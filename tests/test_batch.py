"""Batch mode: many samples, identical parameters, combined outputs."""

from __future__ import annotations

import json

import pytest
from conftest import F, HIGH_Q, LIB_A, LIB_B, LIB_C, LIBRARY, LOW_Q, R, amplicon, fastq_bytes

from psaurus_pcr.batch import fate_breakdown, run_batch
from psaurus_pcr.config import AnalysisParams
from psaurus_pcr.inputs import SampleInput

UNEXPECTED = "GGGGGGGGGGCCCCCCCCCC"


def params():
    return AnalysisParams(max_insert_length=200, r_convention="literal")


def sample(name, records):
    return SampleInput(name=name, source=fastq_bytes(records), origin=f"{name}.fastq")


def three_samples():
    """Deliberately different samples, so cross-sample tables have something to show."""
    return [
        # deep, clean, both library members
        sample("s1", [("r%d" % i, amplicon(LIB_A), HIGH_Q) for i in range(5)]
                     + [("b%d" % i, amplicon(LIB_B), HIGH_Q) for i in range(3)]),
        # shallow, one member only, plus a read that fails QC
        sample("s2", [("r%d" % i, amplicon(LIB_A), HIGH_Q) for i in range(2)]
                     + [("bad", amplicon(LIB_A), LOW_Q)]),
        # contaminated: mostly a sequence that is not in the library
        sample("s3", [("u%d" % i, amplicon(UNEXPECTED), HIGH_Q) for i in range(4)]
                     + [("r0", amplicon(LIB_C), HIGH_Q)]),
    ]


def test_each_sample_is_analysed_independently():
    batch = run_batch(three_samples(), F, R, params=params())

    assert batch.sample_names == ["s1", "s2", "s3"]
    assert batch["s1"].run_summary["quality"]["total_reads"] == 8
    assert batch["s2"].run_summary["quality"]["total_reads"] == 3
    assert batch["s2"].run_summary["quality"]["reads_passing_quality_filter"] == 2
    assert int(batch["s1"].unique_sequences.loc[0, "count"]) == 5
    assert list(batch["s3"].unique_sequences["sequence"]) == [UNEXPECTED, LIB_C]
    # A sample's own result is exactly what a single run would have produced.
    assert batch["s1"].run_summary["parameters"]["max_insert_length_used"] == 200


def test_sample_overview_has_one_row_per_sample():
    batch = run_batch(three_samples(), F, R, library=LIBRARY, params=params())
    overview = batch.sample_overview.set_index("sample")

    assert list(overview.index) == ["s1", "s2", "s3"]
    assert int(overview.loc["s1", "total_reads"]) == 8
    assert int(overview.loc["s1", "reads_extracted"]) == 8
    assert int(overview.loc["s2", "reads_passing_quality"]) == 2
    assert overview.loc["s2", "percent_passing_quality"] == pytest.approx(66.6667, abs=1e-3)
    assert int(overview.loc["s1", "library_sequences_detected"]) == 2
    assert int(overview.loc["s3", "library_sequences_detected"]) == 1
    assert overview.loc["s3", "percent_reads_unexpected"] == 80.0
    assert set(overview["status"]) == {"ok"}


def test_sequence_count_matrix_is_sequence_by_sample():
    batch = run_batch(three_samples(), F, R, library=LIBRARY, params=params())
    matrix = batch.sequence_count_matrix.set_index("sequence")

    assert list(matrix.columns[:3]) == ["length", "s1", "s2"]
    assert int(matrix.loc[LIB_A, "s1"]) == 5
    assert int(matrix.loc[LIB_A, "s2"]) == 2
    assert int(matrix.loc[LIB_A, "s3"]) == 0          # absent means zero, not missing
    assert int(matrix.loc[LIB_A, "total_count"]) == 7
    assert int(matrix.loc[LIB_A, "n_samples_detected"]) == 2
    assert int(matrix.loc[UNEXPECTED, "n_samples_detected"]) == 1
    assert bool(matrix.loc[LIB_A, "in_library"]) is True
    assert bool(matrix.loc[UNEXPECTED, "in_library"]) is False
    # Sorted by total abundance across the batch.
    assert batch.sequence_count_matrix.iloc[0]["sequence"] == LIB_A


def test_combined_tables_are_long_format_with_a_sample_column():
    batch = run_batch(three_samples(), F, R, library=LIBRARY, params=params())

    combined = batch.combined_unique_sequences
    assert combined.columns[0] == "sample"
    assert set(combined["sample"]) == {"s1", "s2", "s3"}
    assert int(combined[(combined["sample"] == "s1") & (combined["sequence"] == LIB_B)]
               ["count"].iloc[0]) == 3

    qc = batch.combined_qc_summary
    assert list(qc.columns) == ["metric", "s1", "s2", "s3"]
    row = qc.set_index("metric").loc["quality.total_reads"]
    assert [int(v) for v in row] == [8, 3, 5]

    library = batch.combined_library_comparison
    assert set(library["sample"]) == {"s1", "s2", "s3"}
    assert len(library) == 3 * len(LIBRARY)


def test_library_detection_matrix_is_member_by_sample():
    batch = run_batch(three_samples(), F, R, library=LIBRARY, params=params())
    matrix = batch.library_detection_matrix.set_index("library_id")
    assert int(matrix.loc["lib_0001", "s1"]) == 5     # LIB_A
    assert int(matrix.loc["lib_0002", "s1"]) == 3     # LIB_B
    assert int(matrix.loc["lib_0002", "s3"]) == 0
    assert int(matrix.loc["lib_0003", "n_samples_detected"]) == 1


def test_library_is_read_once_and_shared(tmp_path, write_library):
    path = write_library(LIBRARY)
    batch = run_batch(three_samples(), F, R, library=path, params=params())
    for result in batch.results.values():
        assert result.run_summary["library"]["library_sequences_unique"] == 3
    assert batch.batch_summary["inputs"]["library_sequences"] == 3


def test_a_failing_sample_does_not_sink_the_batch():
    broken = SampleInput(name="broken", source=b"@only_a_header\nACGT\n", origin="broken.fastq")
    batch = run_batch(three_samples() + [broken], F, R, params=params())

    assert batch.sample_names == ["s1", "s2", "s3"]
    assert len(batch.failures) == 1
    assert batch.failures[0].name == "broken"
    assert batch.batch_summary["samples_analysed"] == 3
    assert batch.batch_summary["samples_failed"] == 1

    overview = batch.sample_overview.set_index("sample")
    assert overview.loc["broken", "status"] == "failed"
    assert overview.loc["broken", "error"]


def test_fail_fast_propagates_the_error():
    broken = SampleInput(name="broken", source=b"@only_a_header\nACGT\n", origin="broken.fastq")
    with pytest.raises(Exception):
        run_batch([broken], F, R, params=params(), continue_on_error=False)


def test_batch_from_a_directory_and_a_zip(write_fastq_dir, write_zip):
    records = [("r%d" % i, amplicon(LIB_A), HIGH_Q) for i in range(3)]
    root = write_fastq_dir({"barcode01.fastq": records, "sub/barcode02.fastq.gz": records})
    archive = write_zip({"order/barcode03.fastq": records})

    batch = run_batch([root, archive], F, R, params=params())
    assert batch.sample_names == ["barcode01", "barcode02", "barcode03"]
    for name in batch.sample_names:
        assert int(batch[name].unique_sequences.loc[0, "count"]) == 3


def test_no_fastq_found_is_a_clear_error(tmp_path):
    (tmp_path / "notes.txt").write_text("nothing")
    with pytest.raises(ValueError, match="No FASTQ files found"):
        run_batch(tmp_path, F, R, params=params())


def test_progress_callback_reports_each_sample():
    seen = []
    run_batch(three_samples(), F, R, params=params(),
              progress_callback=lambda i, n, name: seen.append((i, n, name)))
    assert seen == [(0, 3, "s1"), (1, 3, "s2"), (2, 3, "s3"), (3, 3, "")]


def test_fate_breakdown_is_percentages_per_sample():
    batch = run_batch(three_samples(), F, R, params=params())
    frame = fate_breakdown(batch.results).set_index("sample")
    assert frame.loc["s1", "extracted"] == 100.0
    assert int(frame.loc["s2", "reads"]) == 2


def test_batch_summary_is_json_serialisable():
    batch = run_batch(three_samples(), F, R, library=LIBRARY, params=params())
    restored = json.loads(json.dumps(batch.batch_summary))
    assert restored["samples_analysed"] == 3
    assert restored["totals"]["total_reads"] == 16
    assert restored["parameters"]["min_mean_quality"] == 20.0
    assert set(restored["per_sample"]) == {"s1", "s2", "s3"}
    assert [s["sample"] for s in restored["inputs"]["samples"]] == ["s1", "s2", "s3"]


def test_write_outputs_uses_per_sample_directories_plus_combined_tables(tmp_path):
    batch = run_batch(three_samples(), F, R, library=LIBRARY, params=params())
    out = tmp_path / "batch_out"
    written = batch.write_outputs(out, output_format="tsv", write_per_read=True)

    for name in batch.sample_names:
        assert (out / name / "unique_sequences.tsv").exists()
        assert (out / name / "run_summary.json").exists()
        assert (out / name / "top_sequences.png").exists()
    for stem in (
        "sample_overview", "sequence_count_matrix", "combined_unique_sequences",
        "combined_qc_summary", "combined_library_comparison", "library_detection_matrix",
    ):
        assert (out / f"{stem}.tsv").exists(), stem
    assert (out / "batch_summary.json").exists()
    assert (out / "batch_summary.txt").exists()
    assert (out / "batch_read_fate.png").exists()
    assert (out / "batch_library_detection.png").exists()
    assert "sample_overview" in written and "s1/unique_sequences" in written


def test_single_sample_batch_stays_flat_by_default(tmp_path):
    batch = run_batch([three_samples()[0]], F, R, params=params())
    out = tmp_path / "one"
    batch.write_outputs(out, make_plots=False)
    assert (out / "unique_sequences.tsv").exists()
    assert not (out / "s1").exists()
    assert not (out / "sample_overview.tsv").exists()


def test_single_sample_can_be_forced_into_a_sample_directory(tmp_path):
    batch = run_batch([three_samples()[0]], F, R, params=params())
    out = tmp_path / "one_nested"
    batch.write_outputs(out, make_plots=False, sample_dirs=True)
    assert (out / "s1" / "unique_sequences.tsv").exists()
    assert (out / "sample_overview.tsv").exists()


def test_text_summary_lists_every_sample():
    batch = run_batch(three_samples(), F, R, library=LIBRARY, params=params())
    text = batch.text_summary()
    for name in batch.sample_names:
        assert name in text
    assert "samples analysed  : 3 of 3" in text
    assert "unique across the batch" in text
