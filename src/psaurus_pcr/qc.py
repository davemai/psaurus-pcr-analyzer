"""Step 1 -- per-read quality filtering and read-level QC statistics.

Nanopore basecalls carry a real, non-trivial error rate.  Filtering on the mean
read quality is the cheapest way to remove the long tail of poor reads (pore
blockages, chimeras, short adapter-only fragments) that would otherwise
contribute spurious "unique" sequences to the count table.  We keep the
distributions *before and after* filtering so the user can see whether the
threshold is doing something sensible or decimating the run.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Iterator, List, Tuple

from psaurus_pcr.config import AnalysisParams
from psaurus_pcr.fastq_io import FastqRead
from psaurus_pcr.sequtils import read_quality, summarise_numeric


@dataclass
class QCStats:
    """Read counts and length/quality distributions, before and after filtering."""

    total_reads: int = 0
    passing_reads: int = 0
    lengths_all: List[int] = field(default_factory=list)
    qualities_all: List[float] = field(default_factory=list)
    lengths_pass: List[int] = field(default_factory=list)
    qualities_pass: List[float] = field(default_factory=list)
    # Kept separately (rather than derived) so the QC plots can show the
    # kept/removed split as an exact partition of the run.
    lengths_fail: List[int] = field(default_factory=list)
    qualities_fail: List[float] = field(default_factory=list)

    @property
    def failing_reads(self) -> int:
        return self.total_reads - self.passing_reads

    @property
    def percent_passing(self) -> float:
        if self.total_reads == 0:
            return 0.0
        return 100.0 * self.passing_reads / self.total_reads

    def record(self, read: FastqRead, quality: float, passed: bool) -> None:
        self.total_reads += 1
        self.lengths_all.append(read.length)
        self.qualities_all.append(quality)
        if passed:
            self.passing_reads += 1
            self.lengths_pass.append(read.length)
            self.qualities_pass.append(quality)
        else:
            self.lengths_fail.append(read.length)
            self.qualities_fail.append(quality)

    def to_dict(self) -> dict:
        """JSON-serialisable summary (distributions collapsed to stats)."""
        return {
            "total_reads": self.total_reads,
            "reads_passing_quality_filter": self.passing_reads,
            "reads_failing_quality_filter": self.failing_reads,
            "percent_passing_quality_filter": round(self.percent_passing, 4),
            "read_length_before_filter": summarise_numeric(self.lengths_all),
            "read_length_after_filter": summarise_numeric(self.lengths_pass),
            "read_quality_before_filter": summarise_numeric(self.qualities_all),
            "read_quality_after_filter": summarise_numeric(self.qualities_pass),
        }


def passes_quality(quality: float, min_mean_quality: float) -> bool:
    """A read passes when its summary quality is >= the threshold."""
    return quality >= min_mean_quality


def stream_quality_filter(
    reads: Iterable[FastqRead],
    params: AnalysisParams,
    stats: QCStats,
) -> Iterator[Tuple[FastqRead, float]]:
    """Yield ``(read, quality)`` for reads that pass, updating ``stats`` in place.

    Streaming so that a multi-GB FASTQ never has to be held in memory.
    """
    for read in reads:
        quality = read_quality(read.qualities, params.quality_metric)
        passed = passes_quality(quality, params.min_mean_quality)
        stats.record(read, quality, passed)
        if passed:
            yield read, quality


def filter_reads(
    reads: Iterable[FastqRead],
    params: AnalysisParams | None = None,
) -> Tuple[List[Tuple[FastqRead, float]], QCStats]:
    """Eager convenience wrapper around :func:`stream_quality_filter`.

    Returns the passing ``(read, quality)`` pairs and the populated stats.
    Pure: no file IO, no globals, safe to call from Streamlit or a notebook.
    """
    params = params or AnalysisParams()
    stats = QCStats()
    passing = list(stream_quality_filter(reads, params, stats))
    return passing, stats
