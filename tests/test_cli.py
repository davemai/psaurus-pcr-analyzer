"""The CLI layer: argument plumbing only -- the science is tested elsewhere."""

from __future__ import annotations

import json

import pytest
from conftest import F, HIGH_Q, LIB_A, LIB_B, LIBRARY, LOW_Q, R, amplicon

from psaurus_pcr.cli import build_parser, main, params_from_args


def records():
    return [
        ("a1", amplicon(LIB_A), HIGH_Q),
        ("a2", amplicon(LIB_A), HIGH_Q),
        ("b1", amplicon(LIB_B), HIGH_Q),
        ("bad", amplicon(LIB_A), LOW_Q),
    ]


def test_parser_defaults_match_the_documented_behaviour():
    args = build_parser().parse_args(["reads.fastq", "-F", F, "-R", R])
    params = params_from_args(args)
    assert params.min_mean_quality == 20.0
    assert params.quality_metric == "mean_phred"
    assert params.r_convention == "auto"
    assert params.exclude_ambiguous is False
    assert params.fuzzy_library is False
    assert params.edits_for_flank(F, "forward") == 2  # 12% of 20 bp


def test_cli_runs_and_writes_outputs(tmp_path, write_fastq, write_library, capsys):
    fastq = write_fastq(records())
    library = write_library(LIBRARY)
    outdir = tmp_path / "results"

    code = main([
        str(fastq), "-F", F, "-R", R, "-l", str(library),
        "-o", str(outdir), "--format", "csv", "--per-read", "--no-plots",
    ])
    assert code == 0

    for name in (
        "unique_sequences.csv", "qc_summary.csv", "library_comparison.csv",
        "per_read_classification.csv", "run_summary.json", "run_summary.txt",
    ):
        assert (outdir / name).exists(), name
    assert not list(outdir.glob("*.png")), "--no-plots should suppress figures"

    summary = json.loads((outdir / "run_summary.json").read_text())
    assert summary["quality"]["total_reads"] == 4
    assert summary["quality"]["reads_passing_quality_filter"] == 3
    assert summary["quantification"]["total_extracted_reads"] == 3
    assert summary["library"]["library_sequences_detected"] == 2

    table = (outdir / "unique_sequences.csv").read_text().splitlines()
    assert table[0].startswith("sequence,count,percent_of_extracted")
    assert table[1].startswith(f"{LIB_A},2,")

    printed = capsys.readouterr().out
    assert "Step 1 - quality filtering" in printed
    assert "unique sequences" in printed


def test_cli_plots_are_written_by_default(tmp_path, write_fastq):
    outdir = tmp_path / "plots_run"
    assert main([str(write_fastq(records())), "-F", F, "-R", R, "-o", str(outdir),
                 "--quiet"]) == 0
    for name in (
        "quality_distribution.png", "read_length_distribution.png",
        "insert_length_distribution.png", "top_sequences.png",
    ):
        assert (outdir / name).exists(), name


def test_cli_rejects_a_missing_input_file(tmp_path, capsys):
    with pytest.raises(SystemExit):
        main([str(tmp_path / "nope.fastq"), "-F", F, "-R", R])
    assert "not found" in capsys.readouterr().err


def test_cli_warns_when_nothing_is_extracted(tmp_path, write_fastq, capsys):
    fastq = write_fastq([("r1", "ACGT" * 50, HIGH_Q)])
    code = main([str(fastq), "-F", F, "-R", R, "-o", str(tmp_path / "o"),
                 "--no-plots", "--quiet"])
    assert code == 0
    assert "no inserts were extracted" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# batch mode
# ---------------------------------------------------------------------------


def batch_records(insert):
    return [(f"r{i}", amplicon(insert), HIGH_Q) for i in range(3)]


def test_cli_accepts_several_fastq_files(tmp_path, write_fastq, capsys):
    first = write_fastq(batch_records(LIB_A), name="barcode01.fastq")
    second = write_fastq(batch_records(LIB_B), name="barcode02.fastq")
    outdir = tmp_path / "out"

    assert main([str(first), str(second), "-F", F, "-R", R, "-o", str(outdir),
                 "--no-plots"]) == 0

    assert (outdir / "barcode01" / "unique_sequences.tsv").exists()
    assert (outdir / "barcode02" / "unique_sequences.tsv").exists()
    assert (outdir / "sample_overview.tsv").exists()
    assert (outdir / "sequence_count_matrix.tsv").exists()
    assert (outdir / "batch_summary.json").exists()

    printed = capsys.readouterr().out
    assert "samples analysed  : 2 of 2" in printed
    assert "barcode01" in printed and "barcode02" in printed


def test_cli_accepts_a_directory(tmp_path, write_fastq_dir):
    root = write_fastq_dir({
        "barcode01.fastq": batch_records(LIB_A),
        "nested/barcode02.fastq.gz": batch_records(LIB_B),
    })
    outdir = tmp_path / "dir_out"
    assert main([str(root), "-F", F, "-R", R, "-o", str(outdir),
                 "--no-plots", "--quiet"]) == 0
    assert (outdir / "barcode01" / "unique_sequences.tsv").exists()
    assert (outdir / "barcode02" / "unique_sequences.tsv").exists()


def test_cli_accepts_a_zip_archive(tmp_path, write_zip):
    archive = write_zip({
        "order/barcode01.fastq": batch_records(LIB_A),
        "order/barcode02.fastq": batch_records(LIB_B),
    })
    outdir = tmp_path / "zip_out"
    assert main([str(archive), "-F", F, "-R", R, "-o", str(outdir),
                 "--no-plots", "--quiet"]) == 0
    matrix = (outdir / "sequence_count_matrix.tsv").read_text().splitlines()
    assert matrix[0].split("\t")[:4] == ["sequence", "length", "barcode01", "barcode02"]


def test_cli_list_inputs_reports_without_running(tmp_path, write_fastq_dir, capsys):
    root = write_fastq_dir({
        "barcode01.fastq": batch_records(LIB_A),
        "barcode02.fastq": batch_records(LIB_B),
    })
    outdir = tmp_path / "never_written"
    assert main([str(root), "-F", F, "-R", R, "-o", str(outdir), "--list-inputs"]) == 0
    printed = capsys.readouterr().out
    assert "2 FASTQ file(s) would be analysed" in printed
    assert "barcode01" in printed
    assert not outdir.exists()


def test_cli_pattern_narrows_a_directory_search(tmp_path, write_fastq_dir, capsys):
    root = write_fastq_dir({
        "barcode01.fastq.gz": batch_records(LIB_A),
        "unclassified.fastq.gz": batch_records(LIB_B),
    })
    assert main([str(root), "-F", F, "-R", R, "--list-inputs",
                 "--pattern", "barcode*.fastq.gz"]) == 0
    printed = capsys.readouterr().out
    assert "1 FASTQ file(s)" in printed
    assert "unclassified" not in printed


def test_cli_no_recursive_skips_nested_files(tmp_path, write_fastq_dir, capsys):
    root = write_fastq_dir({
        "barcode01.fastq": batch_records(LIB_A),
        "nested/barcode02.fastq": batch_records(LIB_B),
    })
    assert main([str(root), "-F", F, "-R", R, "--list-inputs", "--no-recursive"]) == 0
    printed = capsys.readouterr().out
    assert "1 FASTQ file(s)" in printed


def test_cli_rejects_no_sample_dirs_for_a_multi_sample_batch(tmp_path, write_fastq, capsys):
    first = write_fastq(batch_records(LIB_A), name="a.fastq")
    second = write_fastq(batch_records(LIB_B), name="b.fastq")
    with pytest.raises(SystemExit):
        main([str(first), str(second), "-F", F, "-R", R, "--no-sample-dirs"])
    assert "needs exactly one sample" in capsys.readouterr().err


def test_cli_errors_when_no_fastq_is_found(tmp_path, capsys):
    (tmp_path / "notes.txt").write_text("nothing")
    assert main([str(tmp_path), "-F", F, "-R", R]) == 2
    assert "no FASTQ files found" in capsys.readouterr().err
