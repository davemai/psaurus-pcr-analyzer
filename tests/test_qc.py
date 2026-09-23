"""Step 1: quality metrics and filtering."""

from __future__ import annotations

import math

from conftest import HIGH_Q, LIB_A, LOW_Q, amplicon, fastq_bytes

from psaurus_pcr.config import AnalysisParams
from psaurus_pcr.fastq_io import read_fastq
from psaurus_pcr.qc import filter_reads
from psaurus_pcr.sequtils import error_probability_quality, mean_phred


def test_mean_phred_is_arithmetic_mean():
    assert mean_phred([20, 20, 20]) == 20
    assert mean_phred([10, 30]) == 20


def test_error_probability_quality_is_stricter_than_mean_phred():
    # A read that is mostly Q40 with a few Q5 bases: the arithmetic mean stays
    # high, but the error-probability qscore correctly collapses.
    qualities = [40] * 90 + [5] * 10
    assert mean_phred(qualities) > 35
    assert error_probability_quality(qualities) < 16
    assert math.isclose(error_probability_quality([20] * 10), 20.0, abs_tol=1e-9)


def test_low_quality_read_is_filtered_out():
    records = [
        ("good_read", amplicon(LIB_A), HIGH_Q),
        ("bad_read", amplicon(LIB_A), LOW_Q),
    ]
    reads = read_fastq(fastq_bytes(records))
    passing, stats = filter_reads(reads, AnalysisParams(min_mean_quality=20.0))

    assert stats.total_reads == 2
    assert stats.passing_reads == 1
    assert stats.failing_reads == 1
    assert stats.percent_passing == 50.0
    assert [read.id for read, _ in passing] == ["good_read"]


def test_threshold_is_inclusive_and_configurable():
    records = [("exactly_q20", amplicon(LIB_A), 20)]
    reads = read_fastq(fastq_bytes(records))

    passing, _ = filter_reads(reads, AnalysisParams(min_mean_quality=20.0))
    assert len(passing) == 1, "a read exactly at the threshold should pass"

    passing, _ = filter_reads(read_fastq(fastq_bytes(records)),
                              AnalysisParams(min_mean_quality=20.5))
    assert passing == []


def test_qc_stats_report_distributions_before_and_after():
    records = [
        ("long_good", amplicon(LIB_A * 5), HIGH_Q),
        ("short_bad", amplicon(LIB_A), LOW_Q),
    ]
    _, stats = filter_reads(read_fastq(fastq_bytes(records)), AnalysisParams())
    summary = stats.to_dict()

    assert summary["total_reads"] == 2
    assert summary["reads_passing_quality_filter"] == 1
    assert summary["read_length_before_filter"]["n"] == 2
    assert summary["read_length_after_filter"]["n"] == 1
    assert summary["read_quality_before_filter"]["min"] == LOW_Q
    assert summary["read_quality_after_filter"]["min"] == HIGH_Q
