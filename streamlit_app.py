"""Streamlit front end for the Plasmidsaurus PCR amplicon analyser.

This file contains *no* analysis logic.  It collects inputs, builds an
:class:`~psaurus_pcr.config.AnalysisParams`, calls
:func:`psaurus_pcr.batch.run_batch`, and renders the DataFrames and figures it
gets back.  That is the whole point of the core/CLI split: the web app and
``psaurus-pcr`` run identical code.

One sample or fifty go through the same path -- ``run_batch`` always returns a
:class:`~psaurus_pcr.batch.BatchResult`, and the UI simply hides the
cross-sample views when there is only one sample.

Run with::

    streamlit run streamlit_app.py
"""

from __future__ import annotations

import io
import json
import os
import sys
from pathlib import Path
from typing import List, Optional, Tuple

# Make ``src/`` importable when the app is run straight from a checkout that has
# not been ``pip install``-ed.
_SRC = Path(__file__).resolve().parent / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import pandas as pd
import streamlit as st

from psaurus_pcr import __version__
from psaurus_pcr.batch import BatchResult, run_batch
from psaurus_pcr.config import (
    DEFAULT_FLANK_EDIT_FRACTION,
    DEFAULT_MIN_MEAN_QUALITY,
    AnalysisParams,
)
from psaurus_pcr.inputs import (
    PathNotAllowed,
    SampleInput,
    is_tar_name,
    list_sample_names,
    resolve_under_root,
    sample_name,
)
from psaurus_pcr.plots import build_all_figures, build_batch_figures
from psaurus_pcr.report import report_bytes

st.set_page_config(
    page_title="Plasmidsaurus PCR analyser",
    page_icon="🧬",
    layout="wide",
)

# ---------------------------------------------------------------------------
# deployment configuration
#
# Read from st.secrets first (Community Cloud's "Secrets" box, or a local
# git-ignored .streamlit/secrets.toml), then the environment.  There are no
# credentials here -- the app talks to nothing but its own filesystem -- but
# these knobs decide how much of that filesystem a visitor can reach, so they
# belong in secrets rather than in committed code.
# ---------------------------------------------------------------------------

#: The repo checkout. On Community Cloud this is where the bundled
#: examples/ folder lives, so a visitor can try the demo data by typing a
#: relative path without being able to read anything else on the container.
APP_ROOT = Path(__file__).resolve().parent


def _setting(name: str, env: str, default=None):
    """A value from st.secrets, else the environment, else ``default``.

    Accessing st.secrets raises when no secrets file exists at all, which is
    the normal case locally, so the lookup is defensive.
    """
    try:
        if name in st.secrets:
            return st.secrets[name]
    except Exception:  # no secrets file configured; fall through
        pass
    return os.environ.get(env, default)


def _truthy(value) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


#: Allow server-side paths outside APP_ROOT.  Off by default: a public
#: deployment must not turn a text box into "read any file on the server".
#: Set `allow_any_path = true` (secrets) or PSAURUS_ALLOW_ANY_PATH=1 on a host
#: where you control who can reach the app -- a lab server or your own laptop.
ALLOW_ANY_PATH = _truthy(_setting("allow_any_path", "PSAURUS_ALLOW_ANY_PATH", False))

#: The directory server-side paths are confined to when ALLOW_ANY_PATH is off.
DATA_ROOT = (
    None if ALLOW_ANY_PATH
    else Path(_setting("data_root", "PSAURUS_DATA_ROOT", APP_ROOT)).expanduser()
)

#: Refuse to start an analysis on more input than the host can hold.  Uploads
#: are parsed in memory and the whole result is cached in memory afterwards,
#: so this is a memory ceiling, not a disk one.  Community Cloud gives each app
#: a modest shared budget; raise this only on a host you control.
MAX_TOTAL_UPLOAD_MB = float(_setting("max_total_upload_mb", "PSAURUS_MAX_UPLOAD_MB", 200))


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _frame_bytes(frame: pd.DataFrame, output_format: str) -> bytes:
    sep = "\t" if output_format == "tsv" else ","
    return frame.to_csv(sep=sep, index=False).encode("utf-8")


#: Rows above which a table is serialised on request rather than on every
#: rerun. st.download_button takes bytes, not a callback, so a direct button
#: re-encodes its whole frame every time the script re-runs -- fine for a QC
#: summary, ruinous for a per-read table with a row per read.
LAZY_DOWNLOAD_ROWS = 5_000

#: Rows sent to the browser for display. The full table is still downloadable;
#: this only stops a 100k-row frame being pushed over the websocket.
MAX_DISPLAY_ROWS = 2_000


def _download(label: str, frame: pd.DataFrame, stem: str, output_format: str, key: str) -> None:
    """A download button, serialising lazily when the table is large."""
    filename = f"{stem}.{output_format}"
    if len(frame) <= LAZY_DOWNLOAD_ROWS:
        st.download_button(
            label, data=_frame_bytes(frame, output_format),
            file_name=filename, mime="text/plain", key=key,
        )
        return

    state_key = f"csv::{key}::{len(frame)}::{output_format}"
    if st.button(f"{label} — prepare ({len(frame):,} rows)", key=f"make::{key}"):
        with st.spinner("Preparing file…"):
            st.session_state["csv_payload"] = (state_key, _frame_bytes(frame, output_format))
    payload = st.session_state.get("csv_payload")
    if payload and payload[0] == state_key:
        st.download_button(
            f"Download {filename}", data=payload[1],
            file_name=filename, mime="text/plain", key=f"dl::{key}",
        )


def _show(frame: pd.DataFrame, **kwargs) -> None:
    """Render a table, capping how many rows go to the browser."""
    if len(frame) > MAX_DISPLAY_ROWS:
        st.caption(
            f"Showing the first {MAX_DISPLAY_ROWS:,} of {len(frame):,} rows. "
            "Filter above to narrow it, or download the full table."
        )
        frame = frame.head(MAX_DISPLAY_ROWS)
    st.dataframe(frame, use_container_width=True, hide_index=True, **kwargs)


# Bounded on purpose: an unbounded st.cache_data is the classic way to walk a
# Community Cloud app into its memory limit, because nothing is ever evicted.
# One hour is long enough to keep a session responsive, and max_entries keeps
# only the report(s) actually being looked at.
@st.cache_data(show_spinner=False, ttl=3600, max_entries=2)
def _pdf_cached(_target, cache_key: Tuple) -> bytes:
    """Render a PDF report, cached on ``cache_key``.

    ``_target`` is underscore-prefixed on purpose: a BatchResult is not
    hashable, so Streamlit must skip it and use ``cache_key`` -- the run
    timestamp, what is being rendered and the top-N setting -- instead.
    Rendering takes seconds, so it must not happen on every script re-run.
    """
    return report_bytes(_target, top_n=cache_key[-1])


def _pdf_download(target, cache_key: Tuple, filename: str, label: str, key: str) -> None:
    """Two-step PDF download: render on request, then offer the file.

    Rendering eagerly would add seconds to every interaction for a report most
    visits never download.
    """
    state_key = f"pdf::{key}::{cache_key}"
    if st.button(label, key=f"make::{key}"):
        with st.spinner("Rendering PDF report…"):
            # One slot, not one key per report: session_state persists for the
            # whole session, so keeping every PDF a user ever rendered would
            # grow without bound.
            st.session_state["pdf_payload"] = (state_key, _pdf_cached(target, cache_key))
    payload = st.session_state.get("pdf_payload")
    if payload and payload[0] == state_key:
        st.download_button(
            f"Download {filename}",
            data=payload[1],
            file_name=filename,
            mime="application/pdf",
            key=f"dl::{key}",
        )


def _path_fingerprint(path: Path) -> Tuple:
    """(path, size, mtime) for every file under ``path``.

    Included in the cache key so that editing or replacing data on disk
    invalidates a cached run instead of silently serving stale results.
    """
    if path.is_dir():
        return tuple(
            (str(p), p.stat().st_size, p.stat().st_mtime)
            for p in sorted(path.rglob("*")) if p.is_file()
        )
    stat = path.stat()
    return ((str(path), stat.st_size, stat.st_mtime),)


def _needs_bytes_to_list(name: str) -> bool:
    """Whether listing this upload's samples requires reading its contents.

    Only archives do: their member names live inside the file.  Everything
    else is one sample named after the upload.
    """
    return name.lower().endswith(".zip") or is_tar_name(name)


def _upload_size(upload) -> int:
    """Byte size of an upload without copying it.

    ``UploadedFile.getvalue()`` materialises a fresh copy, which is a poor way
    to ask how big something is when the answer is already an attribute.
    """
    size = getattr(upload, "size", None)
    if isinstance(size, int):
        return size
    return len(upload.getvalue())


def _named_buffer(name: str, data: bytes) -> io.BytesIO:
    """Wrap uploaded bytes so discovery sees them exactly as it sees a path.

    ``discover_inputs`` derives the sample name and detects archives from the
    object's ``.name``, so an upload routed this way yields ``barcode01`` for
    ``barcode01.fastq.gz`` -- the same label the CLI gives it -- and an
    uploaded zip is expanded with its filename recorded as the container.
    """
    buffer = io.BytesIO(data)
    buffer.name = name
    return buffer


# max_entries=1: a BatchResult holds a per-read table for every sample, so
# caching several runs at once is exactly how this app would get killed.
# Losing the cache (restart, eviction, TTL) is harmless -- the next press of
# "Run analysis" simply recomputes.
@st.cache_data(show_spinner=False, ttl=3600, max_entries=1)
def _run_cached(
    uploads: Tuple[Tuple[str, bytes], ...],
    server_paths: Tuple[str, ...],
    fingerprints: Tuple,
    forward: str,
    reverse: str,
    library_bytes: Optional[bytes],
    params_dict: dict,
    recursive: bool,
    pattern: Optional[str],
) -> BatchResult:
    """Cache on the raw inputs so re-rendering does not re-run the analysis.

    Streamlit reruns the whole script on every widget interaction; without this
    a deep batch would be re-analysed every time you sort a table.

    ``fingerprints`` carries (path, size, mtime) for the server-side inputs and
    exists purely to be part of the cache key -- it must NOT be named with a
    leading underscore, which is how Streamlit marks an argument as unhashable
    and therefore excluded from that key.
    """
    params = AnalysisParams(**params_dict)
    sources: List = [_named_buffer(name, data) for name, data in uploads]
    sources.extend(Path(p) for p in server_paths)
    return run_batch(
        sources=sources,
        forward_flank=forward,
        reverse_flank=reverse,
        library=library_bytes,
        params=params,
        recursive=recursive,
        pattern=pattern or None,
    )


# ---------------------------------------------------------------------------
# sidebar: inputs and parameters
# ---------------------------------------------------------------------------

st.sidebar.title("Inputs")
fastq_uploads = st.sidebar.file_uploader(
    "FASTQ file(s) or an archive",
    type=["fastq", "fq", "gz", "zip", "tar", "tgz", "txt"],
    accept_multiple_files=True,
    help="Upload one or many FASTQ files (plain or gzipped), or a .zip/.tar.gz "
         "of a delivery folder. Every FASTQ found is analysed separately with "
         "the same settings.",
)
_path_help = (
    "A FASTQ file, a directory to search, or an archive, already on the "
    "machine running this app. Use this for folders — browsers cannot upload "
    "a directory. Separate several paths with a comma."
)
if DATA_ROOT is not None:
    _path_help += (
        f" Paths are resolved inside {DATA_ROOT} and may not escape it; "
        "try `examples/batch` for the bundled demo data."
    )
server_path_text = st.sidebar.text_input(
    "…or a path on this machine",
    value="", key="server_path",
    help=_path_help,
).strip()

with st.sidebar.expander("Folder search options"):
    recursive = st.checkbox(
        "Search directories recursively", value=True,
        help="Delivery folders and MinKNOW output are both nested.",
    )
    pattern = st.text_input(
        "Filename glob (optional)", value="",
        help="e.g. barcode*.fastq.gz — applied only when walking a directory.",
    ).strip()

library_upload = st.sidebar.file_uploader(
    "Intended library (optional)", type=["txt", "csv", "tsv", "fa", "fasta"],
    help="One expected sequence per line. '#' comments and '>' FASTA headers "
         "are ignored. The same library is used for every sample.",
)

st.sidebar.markdown("### Flanking sequences (5'→3')")
forward_flank = st.sidebar.text_input(
    "Forward flank (F)", value="", key="forward_flank",
    help="Short sequence immediately 5' of the region of interest.",
).strip()
reverse_flank = st.sidebar.text_input(
    "Reverse flank (R)", value="", key="reverse_flank",
    help="Short sequence flanking the 3' side. See 'R orientation' below.",
).strip()

st.sidebar.markdown("### Step 1 — quality filtering")
min_quality = st.sidebar.slider(
    "Minimum mean read quality", min_value=0.0, max_value=40.0,
    value=float(DEFAULT_MIN_MEAN_QUALITY), step=0.5,
    help="Q20 = 99% accuracy, appropriate for R10.4.1 + 'sup' basecalling "
         "(what Plasmidsaurus runs). Older chemistries were filtered at Q9–Q10.",
)
quality_metric = st.sidebar.selectbox(
    "Quality statistic", options=["mean_phred", "error_prob"], index=0,
    help="mean_phred = arithmetic mean of per-base Phred scores (conventional). "
         "error_prob = -10·log10(mean per-base error probability), how ONT "
         "defines a read qscore; stricter.",
)

st.sidebar.markdown("### Step 2 — flank matching")
flank_edit_fraction = st.sidebar.slider(
    "Allowed edits per flank (fraction of flank length)",
    min_value=0.0, max_value=0.40, value=float(DEFAULT_FLANK_EDIT_FRACTION), step=0.01,
    help="0.12 ⇒ 2 edits for a 20 bp flank. Too loose risks false hits in random "
         "sequence; too tight drops real reads to residual nanopore error.",
)
override_edits = st.sidebar.checkbox("Override with an absolute edit budget", value=False)
forward_max_edits = reverse_max_edits = None
if override_edits:
    forward_max_edits = st.sidebar.number_input("Max edits, F", 0, 20, 2, 1)
    reverse_max_edits = st.sidebar.number_input("Max edits, R", 0, 20, 2, 1)

r_convention = st.sidebar.selectbox(
    "R orientation", options=["auto", "literal", "revcomp"], index=0,
    help="literal: R lies on the same strand as F (5'-F…R-3'). "
         "revcomp: R is a conventional reverse PCR primer, so revcomp(R) is "
         "searched. auto: try both on a subsample and keep the better one. "
         "Resolved per sample.",
)
ambiguity_margin = st.sidebar.number_input(
    "Ambiguity margin (edits)", min_value=0, max_value=10, value=1, step=1,
    help="A second, non-overlapping flank hit within this many edits of the best "
         "flags the read as ambiguous — often a concatemer or read-through.",
)
exclude_ambiguous = st.sidebar.checkbox(
    "Exclude ambiguous reads from counts", value=False,
    help="Off by default: ambiguous reads are counted using their best hit but "
         "flagged and reported separately.",
)

st.sidebar.markdown("### Step 2b — insert sanity checks")
auto_max_insert = st.sidebar.checkbox(
    "Derive max insert length automatically", value=True,
    help="3× the longest library sequence when a library is uploaded, else 5000 bp.",
)
max_insert_length = None
if not auto_max_insert:
    max_insert_length = st.sidebar.number_input(
        "Max insert length (bp)", min_value=1, max_value=1_000_000, value=5000, step=50
    )
min_insert_length = st.sidebar.number_input(
    "Min insert length (bp)", min_value=0, max_value=100_000, value=1, step=1,
    help="Default 1: only truly empty inserts are dropped.",
)

st.sidebar.markdown("### Step 4 — library comparison")
fuzzy_library = st.sidebar.checkbox(
    "Fuzzy library matching", value=False,
    help="Off by default — library members are defined sequences, so a near-miss "
         "is a finding (synthesis or basecall error), not a match.",
)
library_edit_fraction = 0.05
if fuzzy_library:
    library_edit_fraction = st.sidebar.slider(
        "Library fuzzy tolerance (fraction of length)", 0.0, 0.25, 0.05, 0.01
    )

st.sidebar.markdown("### Output")
output_format = st.sidebar.radio("Table format", ["tsv", "csv"], index=0, horizontal=True)
top_n = st.sidebar.number_input("Top N in bar chart", 5, 100, 20, 5)

run_clicked = st.sidebar.button(
    "Run analysis", type="primary", use_container_width=True, key="run",
)


# ---------------------------------------------------------------------------
# main panel
# ---------------------------------------------------------------------------

st.title("Plasmidsaurus PCR amplicon analyser")
st.caption(
    f"v{__version__} · quality-filter nanopore reads, extract the insert between "
    "two fuzzy-matched flanks (both strands searched), collapse and count unique "
    "inserts, and compare them against an intended library. Analyse one FASTQ or "
    "a whole delivery folder — each sample is processed separately with the same "
    "settings."
)

if "batch" not in st.session_state:
    st.session_state.batch = None

# Resolve and confine every server-side path once, here, so the preview, the
# validation and the analysis all act on the same already-checked list and no
# raw user string ever reaches the filesystem.
_raw_paths = [p.strip() for p in server_path_text.split(",") if p.strip()]
resolved_paths: List[Path] = []
path_errors: List[str] = []
for _raw in _raw_paths:
    try:
        resolved_paths.append(resolve_under_root(_raw, DATA_ROOT))
    except (PathNotAllowed, FileNotFoundError, OSError) as exc:
        path_errors.append(str(exc))
server_paths: List[str] = [str(p) for p in resolved_paths]

for _message in path_errors:
    st.sidebar.error(_message)

# Live preview of what would be analysed, before committing to a long run.
if server_paths or fastq_uploads:
    try:
        # list_sample_names avoids decompressing anything: this block re-runs
        # on every widget interaction, so it has to stay cheap.
        preview_sources: List = [
            _named_buffer(upload.name, upload.getvalue())
            if _needs_bytes_to_list(upload.name)
            # A plain FASTQ contributes one sample named after the file, so
            # the preview never has to touch (or copy) its contents.
            else SampleInput(sample_name(upload.name), b"", upload.name)
            for upload in (fastq_uploads or [])
        ]
        preview_sources.extend(resolved_paths)
        preview = list_sample_names(
            preview_sources, recursive=recursive, pattern=pattern or None
        )
        if preview:
            st.info(
                f"**{len(preview)} FASTQ file(s) ready to analyse:** "
                + ", ".join(name for name, _ in preview[:12])
                + (" …" if len(preview) > 12 else "")
            )
        else:
            st.warning(
                "No FASTQ files found in the given input(s). Expected files "
                "ending in .fastq/.fq (optionally .gz), a folder containing "
                "them, or a zip/tar archive of them."
            )
    except (OSError, ValueError) as exc:
        st.warning(f"Could not read one of the inputs: {exc}")

if run_clicked:
    problems = []
    if not fastq_uploads and not server_paths:
        problems.append("upload at least one FASTQ file or give a path on this machine")
    if not forward_flank:
        problems.append("enter the forward flank (F)")
    if not reverse_flank:
        problems.append("enter the reverse flank (R)")
    if path_errors:
        problems.append("fix the input path(s) reported in the sidebar")
    upload_mb = sum(_upload_size(u) for u in (fastq_uploads or [])) / 1e6
    if upload_mb > MAX_TOTAL_UPLOAD_MB:
        problems.append(
            f"upload less at once — {upload_mb:,.0f} MB exceeds this host's "
            f"{MAX_TOTAL_UPLOAD_MB:,.0f} MB limit. Analyse the samples in "
            "smaller batches, or run the `psaurus-pcr` CLI locally, where "
            "nothing has to be held in memory"
        )
    if problems:
        joined = problems[0] if len(problems) == 1 else (
            ", ".join(problems[:-1]) + " and " + problems[-1]
        )
        st.error(f"Before running, please {joined}.")
    else:
        params_dict = AnalysisParams(
            min_mean_quality=float(min_quality),
            quality_metric=quality_metric,
            flank_edit_fraction=float(flank_edit_fraction),
            forward_max_edits=int(forward_max_edits) if override_edits else None,
            reverse_max_edits=int(reverse_max_edits) if override_edits else None,
            ambiguity_margin=int(ambiguity_margin),
            r_convention=r_convention,
            max_insert_length=int(max_insert_length) if max_insert_length else None,
            min_insert_length=int(min_insert_length),
            exclude_ambiguous=bool(exclude_ambiguous),
            fuzzy_library=bool(fuzzy_library),
            library_edit_fraction=float(library_edit_fraction),
            top_n_plot=int(top_n),
        ).to_dict()
        try:
            with st.spinner("Analysing samples…"):
                st.session_state.batch = _run_cached(
                    tuple((u.name, u.getvalue()) for u in (fastq_uploads or [])),
                    tuple(server_paths),
                    tuple(_path_fingerprint(Path(p)) for p in server_paths),
                    forward_flank,
                    reverse_flank,
                    library_upload.getvalue() if library_upload is not None else None,
                    params_dict,
                    bool(recursive),
                    pattern or None,
                )
        except Exception as exc:  # surfaced to the user rather than a stack trace
            st.session_state.batch = None
            st.error(f"Analysis failed: {exc}")

batch: Optional[BatchResult] = st.session_state.batch

if batch is None:
    st.info(
        "Upload one or more FASTQ files (or point at a folder/archive on this "
        "machine), enter the F and R flanking sequences in the sidebar, then "
        "press **Run analysis**."
    )
    st.stop()

if batch.failures:
    st.error(
        f"{len(batch.failures)} sample(s) could not be analysed: "
        + "; ".join(f"**{f.name}** ({f.error})" for f in batch.failures)
    )
if not batch.results:
    st.stop()

info = batch.batch_summary
totals = info["totals"]
is_batch = len(batch.results) > 1

# --- headline numbers -------------------------------------------------------
columns = st.columns(5 if is_batch else 4)
index = 0
if is_batch:
    columns[index].metric("Samples", f"{info['samples_analysed']:,}")
    index += 1
columns[index].metric("Total reads", f"{totals['total_reads']:,}")
columns[index + 1].metric(
    "Passed QC",
    f"{totals['reads_passing_quality']:,}",
    f"{100.0 * totals['reads_passing_quality'] / totals['total_reads']:.1f}%"
    if totals["total_reads"] else "0%",
)
columns[index + 2].metric(
    "Inserts extracted",
    f"{totals['reads_extracted']:,}",
    f"{100.0 * totals['reads_extracted'] / totals['reads_passing_quality']:.1f}% of QC-passing"
    if totals["reads_passing_quality"] else "0%",
)
columns[index + 3].metric(
    "Unique sequences" + (" (batch-wide)" if is_batch else ""),
    f"{totals['unique_sequences_across_batch']:,}",
)

if totals["reads_extracted"] == 0:
    st.warning(
        "No inserts were extracted from any sample. Check the flank sequences, "
        "set the R orientation explicitly, lower the quality threshold, or raise "
        "the allowed edits per flank."
    )

tab_labels = (["Batch overview"] if is_batch else []) + [
    "QC summary", "Unique sequences", "Library comparison", "Plots", "Run summary"
]
tabs = st.tabs(tab_labels)
tab = dict(zip(tab_labels, tabs))

# --- batch overview ---------------------------------------------------------
if is_batch:
    with tab["Batch overview"]:
        st.subheader("Per-sample summary")
        st.dataframe(batch.sample_overview, use_container_width=True, hide_index=True)
        left_dl, right_dl = st.columns(2)
        with left_dl:
            _download(f"Download sample_overview.{output_format}", batch.sample_overview,
                      "sample_overview", output_format, "dl_overview")
        with right_dl:
            _pdf_download(
                batch,
                ("batch", info["timestamp_utc"], int(top_n)),
                "batch_report.pdf",
                "Prepare full PDF report (batch + every sample)",
                "batch",
            )

        left, right = st.columns(2)
        figures = build_batch_figures(batch)
        with left:
            st.pyplot(figures["batch_read_fate"], use_container_width=True)
        with right:
            if "batch_library_detection" in figures:
                st.pyplot(figures["batch_library_detection"], use_container_width=True)
            else:
                st.info("Upload an intended library to see per-sample coverage here.")

        st.subheader("Sequence × sample count matrix")
        st.caption(
            "Every unique insert seen anywhere in the batch, with its count in "
            "each sample. `n_samples_detected` = 1 for a sequence confined to "
            "one sample — often an artefact or index hop."
        )
        matrix = batch.sequence_count_matrix
        m1, m2 = st.columns([2, 1])
        with m1:
            matrix_query = st.text_input(
                "Filter by sequence substring", value="", key="matrix_query"
            ).strip().upper()
        with m2:
            min_total = st.number_input(
                "Minimum total count", min_value=1,
                max_value=int(matrix["total_count"].max()) if not matrix.empty else 1,
                value=1, step=1, key="matrix_min",
            )
        view = matrix
        if matrix_query:
            view = view[view["sequence"].str.contains(matrix_query, regex=False)]
        view = view[view["total_count"] >= min_total]
        _show(view)
        _download(f"Download sequence_count_matrix.{output_format}", matrix,
                  "sequence_count_matrix", output_format, "dl_matrix")

        if batch.library_detection_matrix is not None:
            st.subheader("Library member × sample counts")
            _show(batch.library_detection_matrix)
            _download(f"Download library_detection_matrix.{output_format}",
                      batch.library_detection_matrix, "library_detection_matrix",
                      output_format, "dl_libmatrix")

        with st.expander("Other combined downloads"):
            _download(f"combined_unique_sequences.{output_format}",
                      batch.combined_unique_sequences, "combined_unique_sequences",
                      output_format, "dl_combined_unique")
            _download(f"combined_qc_summary.{output_format}", batch.combined_qc_summary,
                      "combined_qc_summary", output_format, "dl_combined_qc")
            if batch.combined_library_comparison is not None:
                _download(f"combined_library_comparison.{output_format}",
                          batch.combined_library_comparison,
                          "combined_library_comparison", output_format, "dl_combined_lib")
            st.download_button(
                "batch_summary.json",
                data=json.dumps(batch.batch_summary, indent=2, default=str).encode("utf-8"),
                file_name="batch_summary.json", mime="application/json",
                key="dl_batch_json",
            )

# --- per-sample selector ----------------------------------------------------
sample_names = batch.sample_names
if is_batch:
    selected = st.selectbox(
        "Sample shown in the tabs below", options=sample_names, index=0,
        help="Each sample was analysed independently with identical parameters.",
    )
else:
    selected = sample_names[0]
result = batch[selected]
summary = result.run_summary

resolved = summary["inputs"]["r_convention"]
st.caption(
    f"**{selected}** — R interpreted as **{resolved['resolved']}** "
    f"(searched `{summary['inputs']['reverse_flank_searched']}`) — {resolved['note']}"
)

# --- QC ---------------------------------------------------------------------
with tab["QC summary"]:
    st.subheader(f"Read fate — {selected}")
    status_rows = [
        {
            "status": status,
            "reads": count,
            "percent_of_qc_passing": summary["extraction"]["status_percent"].get(status, 0.0),
        }
        for status, count in summary["extraction"]["status_counts"].items()
    ]
    st.dataframe(pd.DataFrame(status_rows), use_container_width=True, hide_index=True)
    st.caption(
        f"{summary['extraction']['reads_with_ambiguous_flank']:,} read(s) had a "
        "second, near-equally-good hit for a flank (possible concatemer or "
        "read-through). These are "
        + ("excluded from" if summary["extraction"]["ambiguous_reads_excluded"] else "included in")
        + " the counts and flagged."
    )

    st.subheader("All QC statistics")
    st.dataframe(result.qc_summary, use_container_width=True, hide_index=True)
    _download(f"Download qc_summary.{output_format} ({selected})", result.qc_summary,
              f"{selected}_qc_summary", output_format, "dl_qc")

# --- unique sequences -------------------------------------------------------
with tab["Unique sequences"]:
    frame = result.unique_sequences
    st.subheader(f"{len(frame):,} unique extracted sequences — {selected}")
    if frame.empty:
        st.info("Nothing was extracted for this sample, so there is no count table.")
    else:
        left, right = st.columns([2, 1])
        with left:
            query = st.text_input(
                "Filter by sequence substring", value="", key="unique_query",
                help="Case-insensitive; leave blank to show everything.",
            ).strip().upper()
        with right:
            min_count = st.number_input(
                "Minimum count", min_value=1,
                max_value=int(frame["count"].max()), value=1, step=1, key="unique_min",
            )
        view = frame
        if query:
            view = view[view["sequence"].str.contains(query, regex=False)]
        view = view[view["count"] >= min_count]
        st.caption(
            f"Showing {len(view):,} of {len(frame):,} rows. Click a column header to sort."
        )
        _show(view)
        _download(f"Download unique_sequences.{output_format} ({selected}, full table)",
                  frame, f"{selected}_unique_sequences", output_format, "dl_unique")

# --- library ----------------------------------------------------------------
with tab["Library comparison"]:
    if result.library_comparison is None:
        st.info("Upload an intended-library file in the sidebar to enable this comparison.")
    else:
        lib = summary["library"]
        l1, l2, l3 = st.columns(3)
        l1.metric(
            "Library members detected",
            f"{lib['library_sequences_detected']:,} / {lib['library_sequences_unique']:,}",
            f"{lib['percent_library_detected']:.1f}%",
        )
        l2.metric(
            "Unexpected unique sequences",
            f"{lib['unexpected_unique_sequences']:,}",
            f"{lib['percent_unique_sequences_unexpected']:.1f}% of unique",
        )
        l3.metric(
            "Reads in unexpected sequences",
            f"{lib['reads_in_unexpected_sequences']:,}",
            f"{lib['percent_reads_unexpected']:.1f}% of extracted",
        )
        st.caption(
            f"Matching mode: **{lib['matching_mode']}**. "
            "Exact matching compares whitespace-stripped, upper-cased sequences; "
            "fuzzy matching additionally allows a global edit budget per library member."
        )
        show_missing = st.checkbox("Show only undetected library members", value=False)
        table = result.library_comparison
        if show_missing:
            table = table[~table["detected"].astype(bool)]
        _show(table)
        _download(f"Download library_comparison.{output_format} ({selected})",
                  result.library_comparison, f"{selected}_library_comparison",
                  output_format, "dl_lib")

# --- plots ------------------------------------------------------------------
with tab["Plots"]:
    figures = build_all_figures(result, top_n=int(top_n))
    left, right = st.columns(2)
    with left:
        st.pyplot(figures["quality_distribution"], use_container_width=True)
        st.pyplot(figures["insert_length_distribution"], use_container_width=True)
    with right:
        st.pyplot(figures["read_length_distribution"], use_container_width=True)
    st.pyplot(figures["top_sequences"], use_container_width=True)

# --- run summary ------------------------------------------------------------
with tab["Run summary"]:
    st.text(result.text_summary())
    st.json(summary, expanded=False)
    st.download_button(
        f"Download run_summary.json ({selected})",
        data=json.dumps(summary, indent=2, default=str).encode("utf-8"),
        file_name=f"{selected}_run_summary.json",
        mime="application/json",
        key="dl_runjson",
    )
    _download(
        f"Download per_read_classification.{output_format} ({selected})",
        result.per_read, f"{selected}_per_read_classification", output_format, "dl_perread",
    )
    st.divider()
    st.caption(
        "A PDF report bundles the parameters, read accounting, tables and every "
        "plot into one file for record keeping."
    )
    _pdf_download(
        result,
        ("sample", summary["timestamp_utc"], selected, int(top_n)),
        f"{selected}_report.pdf",
        f"Prepare PDF report ({selected})",
        f"sample_{selected}",
    )
    if is_batch:
        st.divider()
        st.text(batch.text_summary())
