"""End-to-end orchestration: QC -> extraction -> quantification -> library.

:func:`run_analysis` is *the* entry point for every front end.  It accepts
paths, bytes or file-like objects, never touches ``sys.argv``, never writes a
file unless you ask it to, and returns pandas DataFrames plus a
JSON-serialisable summary dict.  The CLI and ``streamlit_app.py`` are both thin
wrappers around it.
"""

from __future__ import annotations

import datetime as _dt
import os
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd

from psaurus_pcr.config import (
    DEFAULT_MAX_INSERT_LENGTH,
    AnalysisParams,
)
from psaurus_pcr.fastq_io import FastqRead, Source, iter_fastq, read_library
from psaurus_pcr.flanks import (
    ALL_STATUSES,
    STATUS_EXTRACTED,
    ExtractionResult,
    extract_insert,
    orient_reverse_flank,
)
from psaurus_pcr.library import LibraryComparison, compare_to_library, longest_library_sequence
from psaurus_pcr.qc import QCStats, stream_quality_filter
from psaurus_pcr.quantify import quantification_summary, quantify
from psaurus_pcr.sequtils import reverse_complement, validate_flank

ProgressCallback = Optional[Callable[[int], None]]

PER_READ_COLUMNS = [
    "read_id",
    "status",
    "orientation",
    "read_length",
    "mean_quality",
    "insert_length",
    "forward_edit_distance",
    "reverse_edit_distance",
    "ambiguous_flank",
    "counted",
    "insert",
]


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def source_name(source: Source, fallback: str = "<in-memory>") -> str:
    """Best-effort human-readable name for a path / upload / buffer."""
    if isinstance(source, (str, os.PathLike)):
        return str(source)
    name = getattr(source, "name", None)
    if isinstance(name, str) and name:
        return name
    return fallback


def resolve_max_insert_length(
    params: AnalysisParams, library_sequences: Optional[Sequence[str]]
) -> int:
    """Work out the "absurdly long insert" cutoff actually used.

    Precedence: explicit parameter > 3x the longest library member > flat
    default.  Anchoring on the library is the more defensible choice when one
    is available, because it scales with the experiment rather than with a
    guess.
    """
    if params.max_insert_length is not None:
        return int(params.max_insert_length)
    if library_sequences:
        longest = longest_library_sequence(library_sequences)
        if longest > 0:
            return int(round(params.library_length_multiplier * longest))
    return DEFAULT_MAX_INSERT_LENGTH


def _flatten(prefix: str, value, out: Dict[str, object]) -> None:
    if isinstance(value, dict):
        for key, sub in value.items():
            _flatten(f"{prefix}.{key}" if prefix else str(key), sub, out)
    else:
        out[prefix] = value


def flatten_summary(summary: dict) -> pd.DataFrame:
    """Turn the nested run summary into a tidy two-column metric/value table."""
    flat: Dict[str, object] = {}
    _flatten("", summary, flat)
    return pd.DataFrame(
        {"metric": list(flat.keys()), "value": [_stringify(v) for v in flat.values()]}
    )


def _stringify(value):
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return round(value, 6)
    if isinstance(value, (list, tuple)):
        return ";".join(str(v) for v in value)
    return value


# ---------------------------------------------------------------------------
# R-flank convention auto-detection
# ---------------------------------------------------------------------------


@dataclass
class OrientationProbe:
    """Result of deciding how to interpret the user's R sequence."""

    resolved: str                     # "literal" | "revcomp"
    requested: str                    # what the user asked for
    reads_probed: int = 0
    extracted_literal: int = 0
    extracted_revcomp: int = 0
    note: str = ""

    def to_dict(self) -> dict:
        return {
            "requested": self.requested,
            "resolved": self.resolved,
            "reads_probed": self.reads_probed,
            "extracted_with_literal_R": self.extracted_literal,
            "extracted_with_revcomp_R": self.extracted_revcomp,
            "note": self.note,
        }


def detect_r_convention(
    reads: Iterable[Tuple[FastqRead, float]],
    forward_flank: str,
    reverse_flank: str,
    params: AnalysisParams,
    max_insert_length: int,
) -> OrientationProbe:
    """Decide whether R was given on the F strand or as a reverse primer.

    Both conventions are tried on a subsample of quality-passing reads and the
    one that yields more successful F...R extractions wins.  This is cheap
    insurance: "R, 5'->3'" is genuinely ambiguous in the wild -- people write
    reverse *primers* 5'->3' on the bottom strand, but they write the flanking
    sequence of a construct 5'->3' on the top strand.
    """
    literal = reverse_flank
    revcomp = reverse_complement(reverse_flank)

    counts = {"literal": 0, "revcomp": 0}
    probed = 0
    for read, quality in reads:
        probed += 1
        for label, r_seq in (("literal", literal), ("revcomp", revcomp)):
            result = extract_insert(
                read.id,
                read.sequence,
                forward_flank,
                r_seq,
                params,
                max_insert_length,
                read_length=read.length,
                mean_quality=quality,
            )
            if result.status == STATUS_EXTRACTED:
                counts[label] += 1

    if literal == revcomp:
        note = "R is its own reverse complement (palindromic); conventions are equivalent."
        resolved = "literal"
    elif counts["revcomp"] > counts["literal"]:
        resolved = "revcomp"
        note = "revcomp(R) explained more reads; R treated as a reverse PCR primer."
    elif counts["literal"] > counts["revcomp"]:
        resolved = "literal"
        note = "R as typed explained more reads; R treated as lying on the F strand."
    else:
        resolved = "literal"
        note = (
            "Neither convention explained more reads (both "
            f"{counts['literal']}/{probed}); defaulted to R as typed. "
            "If the extraction rate is low, set --r-convention explicitly."
        )

    return OrientationProbe(
        resolved=resolved,
        requested="auto",
        reads_probed=probed,
        extracted_literal=counts["literal"],
        extracted_revcomp=counts["revcomp"],
        note=note,
    )


# ---------------------------------------------------------------------------
# result container
# ---------------------------------------------------------------------------


@dataclass
class AnalysisResult:
    """Structured output of a complete run -- all in memory, nothing on disk."""

    unique_sequences: pd.DataFrame
    qc_summary: pd.DataFrame
    per_read: pd.DataFrame
    run_summary: dict
    library_comparison: Optional[pd.DataFrame] = None
    qc_stats: Optional[QCStats] = None
    insert_lengths: List[int] = field(default_factory=list)

    # -- convenience accessors ---------------------------------------------

    @property
    def status_counts(self) -> Dict[str, int]:
        return dict(self.run_summary["extraction"]["status_counts"])

    def tables(self) -> Dict[str, pd.DataFrame]:
        """Map of output-file stem -> DataFrame (library table omitted if absent)."""
        out = {
            "unique_sequences": self.unique_sequences,
            "qc_summary": self.qc_summary,
        }
        if self.library_comparison is not None:
            out["library_comparison"] = self.library_comparison
        return out

    def text_summary(self) -> str:
        """Compact human-readable run report (what the CLI prints)."""
        s = self.run_summary
        lines = [
            "Plasmidsaurus PCR amplicon analysis",
            f"  run at            : {s['timestamp_local']}",
            f"  fastq             : {s['inputs']['fastq']}",
            f"  F flank           : {s['inputs']['forward_flank']}",
            f"  R flank (given)   : {s['inputs']['reverse_flank_as_given']}",
            f"  R flank (searched): {s['inputs']['reverse_flank_searched']}"
            f"  [{s['inputs']['r_convention']['resolved']}]",
            "",
            "Step 1 - quality filtering",
            f"  total reads       : {s['quality']['total_reads']}",
            f"  passing Q>={s['parameters']['min_mean_quality']:g} ({s['parameters']['quality_metric']})"
            f"    : {s['quality']['reads_passing_quality_filter']}"
            f" ({s['quality']['percent_passing_quality_filter']:.2f}%)",
            "",
            "Step 2 - flank extraction",
        ]
        for status in ALL_STATUSES:
            count = s["extraction"]["status_counts"].get(status, 0)
            pct = s["extraction"]["status_percent"].get(status, 0.0)
            lines.append(f"  {status:<36}: {count} ({pct:.2f}%)")
        lines += [
            f"  reads with an ambiguous flank hit   : "
            f"{s['extraction']['reads_with_ambiguous_flank']}",
            "",
            "Step 3 - quantification",
            f"  extracted reads   : {s['quantification']['total_extracted_reads']}",
            f"  unique sequences  : {s['quantification']['unique_sequences']}",
            f"  singletons        : {s['quantification']['singletons']}"
            f" ({s['quantification']['percent_singletons']:.2f}% of unique)",
        ]
        if s.get("library"):
            lib = s["library"]
            lines += [
                "",
                "Step 4 - library comparison"
                f" ({lib['matching_mode']} matching)",
                f"  library members   : {lib['library_sequences_unique']}",
                f"  detected          : {lib['library_sequences_detected']}"
                f" ({lib['percent_library_detected']:.2f}%)",
                f"  unexpected unique : {lib['unexpected_unique_sequences']}"
                f" ({lib['percent_unique_sequences_unexpected']:.2f}%)",
                f"  unexpected reads  : {lib['reads_in_unexpected_sequences']}"
                f" ({lib['percent_reads_unexpected']:.2f}%)",
            ]
        lines.append("")
        lines.append(f"elapsed: {s['runtime_seconds']:.2f}s")
        return "\n".join(lines)

    # -- persistence --------------------------------------------------------

    def write_outputs(
        self,
        output_dir: os.PathLike | str,
        output_format: str = "tsv",
        write_per_read: bool = False,
        make_plots: bool = True,
        make_pdf: bool = True,
        top_n: int = 20,
    ) -> Dict[str, Path]:
        """Write every output file; returns a map of label -> path written.

        Kept off the analysis path on purpose: nothing downstream reads these
        files back, so a Streamlit caller can skip this entirely and serve the
        same DataFrames through download buttons.
        """
        import json

        from psaurus_pcr.plots import save_all_plots

        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        sep = "\t" if output_format == "tsv" else ","
        ext = output_format
        written: Dict[str, Path] = {}

        for stem, frame in self.tables().items():
            path = out / f"{stem}.{ext}"
            frame.to_csv(path, sep=sep, index=False)
            written[stem] = path

        if write_per_read:
            path = out / f"per_read_classification.{ext}"
            self.per_read.to_csv(path, sep=sep, index=False)
            written["per_read"] = path

        summary_path = out / "run_summary.json"
        summary_path.write_text(json.dumps(self.run_summary, indent=2, default=str))
        written["run_summary_json"] = summary_path

        text_path = out / "run_summary.txt"
        text_path.write_text(self.text_summary() + "\n")
        written["run_summary_txt"] = text_path

        if make_plots:
            written.update(save_all_plots(self, out, top_n=top_n))

        if make_pdf:
            from psaurus_pcr.report import write_sample_report

            pdf_path = out / "report.pdf"
            write_sample_report(self, pdf_path, top_n=top_n)
            written["report_pdf"] = pdf_path

        return written


# ---------------------------------------------------------------------------
# the pipeline
# ---------------------------------------------------------------------------


def run_analysis(
    fastq: Source,
    forward_flank: str,
    reverse_flank: str,
    library: Optional[Source | Sequence[str]] = None,
    params: Optional[AnalysisParams] = None,
    progress_callback: ProgressCallback = None,
) -> AnalysisResult:
    """Run the full pipeline and return everything as in-memory structures.

    Parameters
    ----------
    fastq:
        Path, bytes, or file-like object (e.g. a Streamlit upload).
    forward_flank, reverse_flank:
        5'->3' flank sequences as the user typed them.
    library:
        Optional intended library: a path/bytes/file-like text file with one
        sequence per line, **or** an already-parsed sequence list.
    params:
        :class:`~psaurus_pcr.config.AnalysisParams`; defaults are used if None.
    progress_callback:
        Called with the number of reads processed so far, every 1000 reads.
        Used by the Streamlit progress bar; ignored by the CLI.

    Notes
    -----
    When ``params.r_convention == "auto"`` the FASTQ source is read twice: once
    for a small orientation probe and once for the real pass.  Paths, bytes and
    seekable file objects all support this.  For a non-seekable stream, set
    ``r_convention`` explicitly to ``"literal"`` or ``"revcomp"``.
    """
    started = time.time()
    params = params or AnalysisParams()

    forward = validate_flank(forward_flank, "Forward (F)")
    reverse_given = validate_flank(reverse_flank, "Reverse (R)")

    # --- library (optional) -------------------------------------------------
    library_sequences: Optional[List[str]] = None
    library_label: Optional[str] = None
    if library is not None:
        if isinstance(library, (list, tuple)):
            library_sequences = [str(s) for s in library]
            library_label = "<in-memory list>"
        else:
            library_sequences = read_library(library)
            library_label = source_name(library, "<in-memory library>")

    max_insert_length = resolve_max_insert_length(params, library_sequences)

    # --- resolve the R convention ------------------------------------------
    if params.r_convention == "auto":
        probe_stats = QCStats()
        probe_reads = _take(
            stream_quality_filter(iter_fastq(fastq), params, probe_stats),
            params.orientation_probe_reads,
        )
        probe = detect_r_convention(
            probe_reads, forward, reverse_given, params, max_insert_length
        )
    else:
        probe = OrientationProbe(
            resolved=params.r_convention,
            requested=params.r_convention,
            note="Convention supplied explicitly; no auto-detection performed.",
        )
    reverse_searched = orient_reverse_flank(reverse_given, probe.resolved)

    # --- main pass ----------------------------------------------------------
    qc_stats = QCStats()
    extractions: List[ExtractionResult] = []
    status_counts: Counter = Counter()
    ambiguous_reads = 0
    ambiguous_forward = 0
    ambiguous_reverse = 0
    processed = 0

    for read, quality in stream_quality_filter(iter_fastq(fastq), params, qc_stats):
        result = extract_insert(
            read.id,
            read.sequence,
            forward,
            reverse_searched,
            params,
            max_insert_length,
            read_length=read.length,
            mean_quality=quality,
        )
        extractions.append(result)
        status_counts[result.status] += 1
        if result.forward_ambiguous:
            ambiguous_forward += 1
        if result.reverse_ambiguous:
            ambiguous_reverse += 1
        if result.ambiguous:
            ambiguous_reads += 1
        processed += 1
        if progress_callback is not None and processed % 1000 == 0:
            progress_callback(processed)
    if progress_callback is not None:
        progress_callback(processed)

    # The orientation probe consumed one pass over the source. If that pass
    # saw reads and this one did not, the source could not be rewound -- a
    # pipe or a one-shot stream -- and reporting "0 reads" would be a silent
    # wrong answer. Fail loudly with the fix instead.
    if qc_stats.total_reads == 0 and probe.reads_probed > 0:
        raise ValueError(
            "The FASTQ source could not be read a second time, so no reads "
            "reached the analysis. Auto-detecting the R convention needs two "
            "passes over the input. Either pass a path or bytes instead of a "
            "one-shot stream, or set r_convention to 'literal' or 'revcomp' "
            "explicitly so only one pass is needed."
        )

    # --- step 3 -------------------------------------------------------------
    unique_frame = quantify(extractions)
    quant_summary = quantification_summary(unique_frame)

    # --- step 4 -------------------------------------------------------------
    library_table: Optional[pd.DataFrame] = None
    library_summary: Optional[dict] = None
    if library_sequences is not None:
        comparison: LibraryComparison = compare_to_library(
            unique_frame, library_sequences, params
        )
        unique_frame = comparison.annotated_unique
        library_table = comparison.library_table
        library_summary = comparison.summary

    # --- assemble reporting -------------------------------------------------
    n_qc_pass = qc_stats.passing_reads
    status_percent = {
        status: round(100.0 * status_counts.get(status, 0) / n_qc_pass, 4) if n_qc_pass else 0.0
        for status in ALL_STATUSES
    }

    now = _dt.datetime.now().astimezone()
    run_summary = {
        "tool": "psaurus-pcr-analyzer",
        "version": _version(),
        "timestamp_utc": now.astimezone(_dt.timezone.utc).isoformat(),
        "timestamp_local": now.isoformat(),
        "inputs": {
            "fastq": source_name(fastq),
            "library": library_label,
            "forward_flank": forward,
            "reverse_flank_as_given": reverse_given,
            "reverse_flank_searched": reverse_searched,
            "r_convention": probe.to_dict(),
        },
        "parameters": {
            **params.to_dict(),
            "forward_max_edits_used": params.edits_for_flank(forward, "forward"),
            "reverse_max_edits_used": params.edits_for_flank(reverse_searched, "reverse"),
            "max_insert_length_used": max_insert_length,
        },
        "quality": qc_stats.to_dict(),
        "extraction": {
            "reads_entering_extraction": n_qc_pass,
            "status_counts": {s: int(status_counts.get(s, 0)) for s in ALL_STATUSES},
            "status_percent": status_percent,
            "reads_with_ambiguous_flank": ambiguous_reads,
            "reads_with_ambiguous_forward_flank": ambiguous_forward,
            "reads_with_ambiguous_reverse_flank": ambiguous_reverse,
            "ambiguous_reads_excluded": bool(params.exclude_ambiguous),
        },
        "quantification": quant_summary,
        "library": library_summary,
        "runtime_seconds": round(time.time() - started, 3),
    }

    per_read = pd.DataFrame(
        [r.to_row() for r in extractions], columns=PER_READ_COLUMNS
    )
    insert_lengths = [
        r.insert_length for r in extractions if r.counted and r.insert_length is not None
    ]

    return AnalysisResult(
        unique_sequences=unique_frame,
        qc_summary=flatten_summary(run_summary),
        per_read=per_read,
        run_summary=run_summary,
        library_comparison=library_table,
        qc_stats=qc_stats,
        insert_lengths=insert_lengths,
    )


def _take(iterable, n: int) -> List:
    """First ``n`` items of ``iterable`` as a list (``n <= 0`` means all)."""
    if n is None or n <= 0:
        return list(iterable)
    out = []
    for item in iterable:
        out.append(item)
        if len(out) >= n:
            break
    return out


def _version() -> str:
    from psaurus_pcr import __version__

    return __version__
