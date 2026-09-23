"""Run the same analysis over many FASTQ samples and combine the results.

The unit of work is unchanged: every sample goes through
:func:`psaurus_pcr.pipeline.run_analysis` on its own, with identical
parameters, and keeps its own complete result.  What this module adds is the
cross-sample view -- a per-sample overview table, a sequence x sample count
matrix, and combined QC / library tables -- which is what you actually want
when comparing barcodes from one run or replicates across runs.

A sample that fails (truncated file, wrong format) is recorded as a failure
and the batch carries on; losing the whole run because one barcode is corrupt
would be the wrong trade.
"""

from __future__ import annotations

import datetime as _dt
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Union

import pandas as pd

from psaurus_pcr.config import AnalysisParams
from psaurus_pcr.fastq_io import Source
from psaurus_pcr.flanks import (
    ALL_STATUSES,
    STATUS_AMBIGUOUS_EXCLUDED,
    STATUS_EMPTY_INSERT,
    STATUS_EXTRACTED,
    STATUS_NO_FLANKS,
    STATUS_NO_FORWARD,
    STATUS_NO_REVERSE,
    STATUS_TOO_LONG,
    STATUS_TOO_SHORT,
    STATUS_WRONG_ORDER,
)
from psaurus_pcr.inputs import SampleInput, discover_inputs
from psaurus_pcr.pipeline import AnalysisResult, run_analysis, source_name


def _version() -> str:
    from psaurus_pcr import __version__

    return __version__

#: ``callback(index, total, sample_name)`` -- called before each sample runs.
BatchProgressCallback = Optional[Callable[[int, int, str], None]]

#: Read fates grouped into a handful of buckets for the cross-sample plot and
#: the overview table.  The detailed nine-status breakdown is still in every
#: sample's own QC summary.
FATE_GROUPS = {
    "extracted": (STATUS_EXTRACTED,),
    "flank not found": (STATUS_NO_FLANKS, STATUS_NO_FORWARD, STATUS_NO_REVERSE),
    "wrong order / overlap": (STATUS_WRONG_ORDER,),
    "insert length rejected": (STATUS_EMPTY_INSERT, STATUS_TOO_SHORT, STATUS_TOO_LONG),
    "ambiguous excluded": (STATUS_AMBIGUOUS_EXCLUDED,),
}


@dataclass(frozen=True)
class SampleFailure:
    """A sample that could not be analysed, kept so it is never lost silently."""

    name: str
    origin: str
    error: str

    def to_dict(self) -> dict:
        return {"sample": self.name, "origin": self.origin, "error": self.error}


# ---------------------------------------------------------------------------
# combined-table builders (pure functions over a {name: AnalysisResult} map)
# ---------------------------------------------------------------------------


def build_sample_overview(
    results: Dict[str, AnalysisResult],
    samples: Sequence[SampleInput] = (),
    failures: Sequence[SampleFailure] = (),
) -> pd.DataFrame:
    """One row per sample: the headline numbers you scan down a batch for."""
    origins = {s.name: s.describe() for s in samples}
    rows: List[dict] = []
    for name, result in results.items():
        summary = result.run_summary
        quality = summary["quality"]
        extraction = summary["extraction"]
        quant = summary["quantification"]
        counts = extraction["status_counts"]
        row = {
            "sample": name,
            "status": "ok",
            "origin": origins.get(name, summary["inputs"]["fastq"]),
            "total_reads": quality["total_reads"],
            "reads_passing_quality": quality["reads_passing_quality_filter"],
            "percent_passing_quality": quality["percent_passing_quality_filter"],
            "reads_extracted": quant["total_extracted_reads"],
            "percent_extracted_of_qc": extraction["status_percent"].get(STATUS_EXTRACTED, 0.0),
            "percent_extracted_of_total": round(
                100.0 * quant["total_extracted_reads"] / quality["total_reads"], 4
            )
            if quality["total_reads"]
            else 0.0,
            "unique_sequences": quant["unique_sequences"],
            "singletons": quant["singletons"],
            "percent_singletons": quant["percent_singletons"],
            "median_insert_length": quant["insert_length"]["median"],
            "top_sequence_count": quant["top_sequence_count"],
            "percent_reads_in_top_sequence": quant["percent_reads_in_top_sequence"],
            "reads_with_ambiguous_flank": extraction["reads_with_ambiguous_flank"],
            "r_convention": summary["inputs"]["r_convention"]["resolved"],
            "runtime_seconds": summary["runtime_seconds"],
        }
        for status in ALL_STATUSES:
            row[f"reads_{status}"] = int(counts.get(status, 0))
        if summary.get("library"):
            lib = summary["library"]
            row.update(
                {
                    "library_sequences_detected": lib["library_sequences_detected"],
                    "percent_library_detected": lib["percent_library_detected"],
                    "unexpected_unique_sequences": lib["unexpected_unique_sequences"],
                    "percent_reads_unexpected": lib["percent_reads_unexpected"],
                }
            )
        rows.append(row)

    for failure in failures:
        rows.append({"sample": failure.name, "status": "failed",
                     "origin": failure.origin, "error": failure.error})

    frame = pd.DataFrame(rows)
    if not frame.empty and "error" not in frame.columns:
        frame["error"] = None
    return frame


def build_combined_unique_sequences(results: Dict[str, AnalysisResult]) -> pd.DataFrame:
    """Long-format concatenation of every sample's count table."""
    frames = []
    for name, result in results.items():
        if result.unique_sequences.empty:
            continue
        frame = result.unique_sequences.copy()
        frame.insert(0, "sample", name)
        frames.append(frame)
    if not frames:
        return pd.DataFrame(columns=["sample", "sequence", "count",
                                     "percent_of_extracted", "length"])
    return pd.concat(frames, ignore_index=True)


def build_sequence_count_matrix(results: Dict[str, AnalysisResult]) -> pd.DataFrame:
    """Sequence x sample matrix of raw counts -- the main comparison table.

    Rows are every unique insert seen anywhere in the batch; columns are the
    samples (absent = 0).  ``total_count`` and ``n_samples_detected`` make it
    easy to spot sequences that are abundant overall versus ones confined to a
    single sample (a hallmark of index hopping or a sample-specific artefact).
    """
    long = build_combined_unique_sequences(results)
    if long.empty:
        return pd.DataFrame(columns=["sequence", "length", "total_count",
                                     "n_samples_detected"])

    matrix = (
        long.pivot_table(index="sequence", columns="sample", values="count",
                         aggfunc="sum", fill_value=0)
        .reindex(columns=list(results.keys()), fill_value=0)
    )
    matrix.columns.name = None
    matrix = matrix.astype("int64").reset_index()

    lengths = long.drop_duplicates("sequence").set_index("sequence")["length"]
    sample_columns = [name for name in results.keys() if name in matrix.columns]
    matrix.insert(1, "length", matrix["sequence"].map(lengths).astype("int64"))
    matrix["total_count"] = matrix[sample_columns].sum(axis=1)
    matrix["n_samples_detected"] = (matrix[sample_columns] > 0).sum(axis=1)

    # Carry library annotation across if any sample had one; membership is a
    # property of the sequence, so it is identical wherever it was observed.
    if "in_library" in long.columns:
        annotation = (
            long.dropna(subset=["library_match_type"])
            .drop_duplicates("sequence")
            .set_index("sequence")
        )
        matrix["library_id"] = matrix["sequence"].map(annotation["library_id"])
        matrix["library_match_type"] = (
            matrix["sequence"].map(annotation["library_match_type"]).fillna("none")
        )
        matrix["in_library"] = matrix["library_match_type"] != "none"

    return matrix.sort_values(
        ["total_count", "sequence"], ascending=[False, True], kind="mergesort"
    ).reset_index(drop=True)


def build_combined_qc_summary(results: Dict[str, AnalysisResult]) -> pd.DataFrame:
    """Wide metric x sample table, for eyeballing a batch in one glance."""
    combined: Optional[pd.DataFrame] = None
    for name, result in results.items():
        frame = result.qc_summary.rename(columns={"value": name})
        combined = frame if combined is None else combined.merge(frame, on="metric", how="outer")
    if combined is None:
        return pd.DataFrame(columns=["metric"])
    return combined


def build_combined_library_comparison(
    results: Dict[str, AnalysisResult]
) -> Optional[pd.DataFrame]:
    frames = []
    for name, result in results.items():
        if result.library_comparison is None:
            continue
        frame = result.library_comparison.copy()
        frame.insert(0, "sample", name)
        frames.append(frame)
    if not frames:
        return None
    return pd.concat(frames, ignore_index=True)


def build_library_detection_matrix(
    results: Dict[str, AnalysisResult]
) -> Optional[pd.DataFrame]:
    """Library member x sample matrix of read counts."""
    combined = build_combined_library_comparison(results)
    if combined is None or combined.empty:
        return None
    matrix = (
        combined.pivot_table(index=["library_id", "library_sequence"],
                             columns="sample", values="count",
                             aggfunc="sum", fill_value=0)
        .reindex(columns=[n for n in results if n in set(combined["sample"])], fill_value=0)
    )
    matrix.columns.name = None
    matrix = matrix.astype("int64").reset_index()
    sample_columns = [c for c in matrix.columns if c not in ("library_id", "library_sequence")]
    matrix["total_count"] = matrix[sample_columns].sum(axis=1)
    matrix["n_samples_detected"] = (matrix[sample_columns] > 0).sum(axis=1)
    return matrix.sort_values("library_id", kind="mergesort").reset_index(drop=True)


def fate_breakdown(results: Dict[str, AnalysisResult]) -> pd.DataFrame:
    """Per-sample read fates collapsed into :data:`FATE_GROUPS`, as percentages."""
    rows = []
    for name, result in results.items():
        counts = result.run_summary["extraction"]["status_counts"]
        total = sum(counts.values())
        row = {"sample": name, "reads": total}
        for label, statuses in FATE_GROUPS.items():
            n = sum(int(counts.get(s, 0)) for s in statuses)
            row[label] = round(100.0 * n / total, 4) if total else 0.0
        rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# result container
# ---------------------------------------------------------------------------


@dataclass
class BatchResult:
    """Every per-sample result plus the combined cross-sample tables."""

    results: Dict[str, AnalysisResult]
    failures: List[SampleFailure]
    samples: List[SampleInput]
    batch_summary: dict
    sample_overview: pd.DataFrame
    combined_unique_sequences: pd.DataFrame
    sequence_count_matrix: pd.DataFrame
    combined_qc_summary: pd.DataFrame
    combined_library_comparison: Optional[pd.DataFrame] = None
    library_detection_matrix: Optional[pd.DataFrame] = None

    @property
    def sample_names(self) -> List[str]:
        return list(self.results.keys())

    def __getitem__(self, name: str) -> AnalysisResult:
        return self.results[name]

    def tables(self) -> Dict[str, pd.DataFrame]:
        """Combined output tables, keyed by file stem."""
        out = {
            "sample_overview": self.sample_overview,
            "combined_unique_sequences": self.combined_unique_sequences,
            "sequence_count_matrix": self.sequence_count_matrix,
            "combined_qc_summary": self.combined_qc_summary,
        }
        if self.combined_library_comparison is not None:
            out["combined_library_comparison"] = self.combined_library_comparison
        if self.library_detection_matrix is not None:
            out["library_detection_matrix"] = self.library_detection_matrix
        return out

    def text_summary(self) -> str:
        info = self.batch_summary
        lines = [
            "Plasmidsaurus PCR amplicon analysis - batch",
            f"  run at            : {info['timestamp_local']}",
            f"  samples analysed  : {info['samples_analysed']}"
            f" of {info['samples_discovered']} discovered",
            f"  F flank           : {info['inputs']['forward_flank']}",
            f"  R flank (given)   : {info['inputs']['reverse_flank_as_given']}",
            f"  library           : {info['inputs']['library'] or '(none)'}",
            "",
        ]
        if self.sample_overview.empty:
            lines.append("  no samples produced results")
        else:
            header = (
                f"  {'sample':<24}{'reads':>10}{'passQC%':>9}"
                f"{'extract%':>10}{'unique':>8}{'top%':>7}"
            )
            has_library = "percent_library_detected" in self.sample_overview.columns
            if has_library:
                header += f"{'lib%':>7}{'unexp%':>8}"
            lines += [header, "  " + "-" * (len(header) - 2)]
            for _, row in self.sample_overview.iterrows():
                if row.get("status") != "ok":
                    lines.append(f"  {str(row['sample'])[:24]:<24}   FAILED: {row.get('error')}")
                    continue
                line = (
                    f"  {str(row['sample'])[:24]:<24}"
                    f"{int(row['total_reads']):>10,}"
                    f"{row['percent_passing_quality']:>9.1f}"
                    f"{row['percent_extracted_of_qc']:>10.1f}"
                    f"{int(row['unique_sequences']):>8,}"
                    f"{row['percent_reads_in_top_sequence']:>7.1f}"
                )
                if has_library:
                    line += (
                        f"{row.get('percent_library_detected', float('nan')):>7.1f}"
                        f"{row.get('percent_reads_unexpected', float('nan')):>8.1f}"
                    )
                lines.append(line)

        totals = info["totals"]
        lines += [
            "",
            f"  total reads       : {totals['total_reads']:,}",
            f"  total extracted   : {totals['reads_extracted']:,}",
            f"  sequences seen    : {totals['unique_sequences_across_batch']:,}"
            " unique across the batch",
        ]
        if self.failures:
            lines.append(f"  failed samples    : {len(self.failures)}")
            for failure in self.failures:
                lines.append(f"    - {failure.name}: {failure.error}")
        lines.append("")
        lines.append(f"elapsed: {info['runtime_seconds']:.2f}s")
        return "\n".join(lines)

    # -- persistence --------------------------------------------------------

    def write_outputs(
        self,
        output_dir: Union[str, os.PathLike],
        output_format: str = "tsv",
        write_per_read: bool = False,
        make_plots: bool = True,
        make_pdf: bool = True,
        top_n: int = 20,
        sample_dirs: Optional[bool] = None,
    ) -> Dict[str, Path]:
        """Write per-sample outputs and the combined batch tables.

        ``sample_dirs`` controls layout: ``True`` puts each sample in
        ``<outdir>/<sample>/``, ``False`` writes a single sample's files
        straight into ``<outdir>``.  ``None`` (the default) picks ``False``
        for a one-sample batch and ``True`` otherwise, so the single-file case
        stays flat and familiar.
        """
        from psaurus_pcr.plots import save_batch_plots

        if sample_dirs is None:
            sample_dirs = len(self.results) != 1
        if not sample_dirs and len(self.results) > 1:
            raise ValueError(
                f"sample_dirs=False needs exactly one sample, but this batch has "
                f"{len(self.results)}; without per-sample directories they would "
                "overwrite each other."
            )
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        sep = "\t" if output_format == "tsv" else ","
        written: Dict[str, Path] = {}

        for name, result in self.results.items():
            target = out / name if sample_dirs else out
            sample_files = result.write_outputs(
                target, output_format=output_format,
                write_per_read=write_per_read, make_plots=make_plots,
                # A batch gets one combined PDF covering every sample, so the
                # per-sample PDFs would be pure duplication.
                make_pdf=make_pdf and not sample_dirs,
                top_n=top_n,
            )
            for label, path in sample_files.items():
                written[f"{name}/{label}"] = path

        if sample_dirs:
            for stem, frame in self.tables().items():
                path = out / f"{stem}.{output_format}"
                frame.to_csv(path, sep=sep, index=False)
                written[stem] = path

            summary_path = out / "batch_summary.json"
            summary_path.write_text(json.dumps(self.batch_summary, indent=2, default=str))
            written["batch_summary_json"] = summary_path

            text_path = out / "batch_summary.txt"
            text_path.write_text(self.text_summary() + "\n")
            written["batch_summary_txt"] = text_path

            if make_plots and self.results:
                written.update(save_batch_plots(self, out))

            if make_pdf and self.results:
                from psaurus_pcr.report import write_batch_report

                pdf_path = out / "batch_report.pdf"
                write_batch_report(self, pdf_path, top_n=top_n)
                written["batch_report_pdf"] = pdf_path

        return written


# ---------------------------------------------------------------------------
# the batch runner
# ---------------------------------------------------------------------------


def run_batch(
    sources: Union[Source, SampleInput, Iterable],
    forward_flank: str,
    reverse_flank: str,
    library: Optional[Union[Source, Sequence[str]]] = None,
    params: Optional[AnalysisParams] = None,
    recursive: bool = True,
    pattern: Optional[str] = None,
    progress_callback: BatchProgressCallback = None,
    continue_on_error: bool = True,
) -> BatchResult:
    """Analyse every FASTQ found in ``sources`` with identical parameters.

    ``sources`` may be a single path, a directory, a ``.zip``/``.tar.gz``
    archive, raw bytes, a Streamlit upload, or any mix of those in a list --
    see :func:`psaurus_pcr.inputs.discover_inputs`.

    The library file, if given, is read once and shared across samples, so a
    file-like object that can only be read a single time still works.
    """
    started = time.time()
    params = params or AnalysisParams()
    samples = discover_inputs(sources, recursive=recursive, pattern=pattern)
    if not samples:
        raise ValueError(
            "No FASTQ files found in the given input(s). Expected files ending "
            "in .fastq/.fq (optionally .gz), a directory containing them, or a "
            "zip/tar archive of them."
        )

    # Materialise the library once: reading it per sample would exhaust a
    # one-shot stream and would re-parse the same file N times.
    library_sequences: Optional[List[str]] = None
    library_label: Optional[str] = None
    if library is not None:
        if isinstance(library, (list, tuple)):
            library_sequences = [str(s) for s in library]
            library_label = "<in-memory list>"
        else:
            from psaurus_pcr.fastq_io import read_library

            library_sequences = read_library(library)
            library_label = source_name(library, "<in-memory library>")

    results: Dict[str, AnalysisResult] = {}
    failures: List[SampleFailure] = []
    total = len(samples)
    for index, sample in enumerate(samples):
        if progress_callback is not None:
            progress_callback(index, total, sample.name)
        try:
            results[sample.name] = run_analysis(
                fastq=sample.source,
                forward_flank=forward_flank,
                reverse_flank=reverse_flank,
                library=library_sequences,
                params=params,
            )
        except Exception as exc:  # one bad file must not sink the batch
            if not continue_on_error:
                raise
            failures.append(
                SampleFailure(name=sample.name, origin=sample.describe(), error=f"{type(exc).__name__}: {exc}")
            )
    if progress_callback is not None:
        progress_callback(total, total, "")

    overview = build_sample_overview(results, samples, failures)
    combined_unique = build_combined_unique_sequences(results)
    matrix = build_sequence_count_matrix(results)

    now = _dt.datetime.now().astimezone()
    totals = {
        "total_reads": int(sum(r.run_summary["quality"]["total_reads"] for r in results.values())),
        "reads_passing_quality": int(
            sum(r.run_summary["quality"]["reads_passing_quality_filter"] for r in results.values())
        ),
        "reads_extracted": int(
            sum(r.run_summary["quantification"]["total_extracted_reads"] for r in results.values())
        ),
        "unique_sequences_across_batch": int(len(matrix)),
    }
    batch_summary = {
        "tool": "psaurus-pcr-analyzer",
        "version": _version(),
        "mode": "batch",
        "timestamp_utc": now.astimezone(_dt.timezone.utc).isoformat(),
        "timestamp_local": now.isoformat(),
        "inputs": {
            "forward_flank": forward_flank,
            "reverse_flank_as_given": reverse_flank,
            "library": library_label,
            "library_sequences": len(library_sequences) if library_sequences else 0,
            "samples": [
                {"sample": s.name, "origin": s.origin, "container": s.container}
                for s in samples
            ],
        },
        "parameters": params.to_dict(),
        "samples_discovered": total,
        "samples_analysed": len(results),
        "samples_failed": len(failures),
        "failures": [f.to_dict() for f in failures],
        "totals": totals,
        "per_sample": {name: r.run_summary for name, r in results.items()},
        "runtime_seconds": round(time.time() - started, 3),
    }

    return BatchResult(
        results=results,
        failures=failures,
        samples=samples,
        batch_summary=batch_summary,
        sample_overview=overview,
        combined_unique_sequences=combined_unique,
        sequence_count_matrix=matrix,
        combined_qc_summary=build_combined_qc_summary(results),
        combined_library_comparison=build_combined_library_comparison(results),
        library_detection_matrix=build_library_detection_matrix(results),
    )
