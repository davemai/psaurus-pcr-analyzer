"""End-to-end behaviour of run_analysis, plus the CLI wrapper."""

from __future__ import annotations

import json

from conftest import (
    F,
    HIGH_Q,
    LIB_A,
    LIB_B,
    LIB_C,
    LIBRARY,
    LOW_Q,
    PAD_LEFT,
    PAD_RIGHT,
    R,
    R_AS_PRIMER,
    amplicon,
    fastq_bytes,
    mutate,
)

from psaurus_pcr.config import AnalysisParams
from psaurus_pcr.pipeline import resolve_max_insert_length, run_analysis
from psaurus_pcr.sequtils import reverse_complement

UNEXPECTED = "GGGGGGGGGGCCCCCCCCCC"


def mixed_records():
    """One record of every situation the pipeline is supposed to distinguish."""
    unit = F + LIB_A + R
    return [
        # 3 clean LIB_A reads, one of them reverse-complemented
        ("clean_1", amplicon(LIB_A), HIGH_Q),
        ("clean_2", amplicon(LIB_A), HIGH_Q),
        ("revcomp_1", reverse_complement(amplicon(LIB_A)), HIGH_Q),
        # fuzzy flanks, still LIB_A
        ("fuzzy_1", PAD_LEFT + mutate(F, [3, 11]) + LIB_A + mutate(R, [5]) + PAD_RIGHT, HIGH_Q),
        # a second library member
        ("clean_b", amplicon(LIB_B), HIGH_Q),
        # something not in the library
        ("unexpected", amplicon(UNEXPECTED), HIGH_Q),
        # edge cases, all excluded from counting but reported
        ("no_reverse", PAD_LEFT + F + LIB_A + PAD_RIGHT, HIGH_Q),
        ("no_forward", PAD_LEFT + LIB_A + R + PAD_RIGHT, HIGH_Q),
        ("wrong_order", PAD_LEFT + R + LIB_A + F + PAD_RIGHT, HIGH_Q),
        ("empty_insert", PAD_LEFT + F + R + PAD_RIGHT, HIGH_Q),
        ("too_long", amplicon("ACGT" * 200), HIGH_Q),
        ("concatemer", PAD_LEFT + unit + "GGATCCTT" + unit + PAD_RIGHT, HIGH_Q),
        # and a read that quality filtering should remove before any of this
        ("low_quality", amplicon(LIB_A), LOW_Q),
    ]


def base_params(**kwargs):
    defaults = dict(max_insert_length=200, r_convention="literal")
    defaults.update(kwargs)
    return AnalysisParams(**defaults)


def test_end_to_end_read_accounting():
    result = run_analysis(
        fastq_bytes(mixed_records()), F, R, library=LIBRARY, params=base_params()
    )
    summary = result.run_summary

    assert summary["quality"]["total_reads"] == 13
    assert summary["quality"]["reads_passing_quality_filter"] == 12
    assert summary["quality"]["percent_passing_quality_filter"] > 92

    counts = summary["extraction"]["status_counts"]
    # clean x2 + revcomp + fuzzy + concatemer -> LIB_A ; plus LIB_B and UNEXPECTED
    assert counts["extracted"] == 7
    assert counts["forward_flank_not_found"] == 1
    assert counts["reverse_flank_not_found"] == 1
    assert counts["flanks_wrong_order_or_overlapping"] == 1
    assert counts["insert_empty"] == 1
    assert counts["insert_too_long"] == 1
    assert sum(counts.values()) == 12

    assert summary["extraction"]["reads_with_ambiguous_flank"] == 1
    assert summary["extraction"]["ambiguous_reads_excluded"] is False


def test_end_to_end_quantification_and_library():
    result = run_analysis(
        fastq_bytes(mixed_records()), F, R, library=LIBRARY, params=base_params()
    )
    frame = result.unique_sequences
    # Ties in count are broken by sequence, ascending, so the table is stable
    # between runs: UNEXPECTED ("GGGG...") sorts before LIB_B ("TTTT...").
    assert list(frame["sequence"]) == [LIB_A, UNEXPECTED, LIB_B]
    assert list(frame["count"]) == [5, 1, 1]
    assert bool(frame.loc[0, "in_library"]) is True
    assert bool(frame.loc[1, "in_library"]) is False
    assert bool(frame.loc[2, "in_library"]) is True

    lib = result.run_summary["library"]
    assert lib["library_sequences_unique"] == 3
    assert lib["library_sequences_detected"] == 2
    assert lib["unexpected_unique_sequences"] == 1
    assert lib["reads_in_unexpected_sequences"] == 1

    table = result.library_comparison.set_index("library_id")
    assert int(table.loc["lib_0001", "count"]) == 5
    assert bool(table.loc["lib_0003", "detected"]) is False
    assert LIB_C not in list(frame["sequence"])


def test_excluding_ambiguous_reads_changes_the_counts():
    result = run_analysis(
        fastq_bytes(mixed_records()), F, R, params=base_params(exclude_ambiguous=True)
    )
    counts = result.run_summary["extraction"]["status_counts"]
    assert counts["ambiguous_flank_excluded"] == 1
    assert counts["extracted"] == 6
    assert int(result.unique_sequences.loc[0, "count"]) == 4


def test_per_read_table_has_one_row_per_qc_passing_read():
    result = run_analysis(fastq_bytes(mixed_records()), F, R, params=base_params())
    per_read = result.per_read
    assert len(per_read) == 12
    assert "low_quality" not in set(per_read["read_id"])
    row = per_read.set_index("read_id").loc["revcomp_1"]
    assert row["orientation"] == "reverse"
    assert row["insert"] == LIB_A
    assert bool(per_read.set_index("read_id").loc["concatemer", "ambiguous_flank"]) is True


# --- R-convention auto-detection -------------------------------------------


def test_auto_detects_a_literal_r_flank():
    records = [("r%d" % i, amplicon(LIB_A), HIGH_Q) for i in range(5)]
    result = run_analysis(fastq_bytes(records), F, R, params=AnalysisParams())
    probe = result.run_summary["inputs"]["r_convention"]
    assert probe["resolved"] == "literal"
    assert probe["extracted_with_literal_R"] == 5
    assert probe["extracted_with_revcomp_R"] == 0
    assert result.run_summary["inputs"]["reverse_flank_searched"] == R
    assert int(result.unique_sequences.loc[0, "count"]) == 5


def test_auto_detects_a_reverse_primer_style_r_flank():
    """R typed 5'->3' as a wet-lab reverse primer must still work."""
    records = [("r%d" % i, amplicon(LIB_A), HIGH_Q) for i in range(5)]
    result = run_analysis(fastq_bytes(records), F, R_AS_PRIMER, params=AnalysisParams())
    probe = result.run_summary["inputs"]["r_convention"]
    assert probe["resolved"] == "revcomp"
    assert result.run_summary["inputs"]["reverse_flank_searched"] == R
    assert int(result.unique_sequences.loc[0, "count"]) == 5


def test_explicit_convention_skips_the_probe():
    records = [("r1", amplicon(LIB_A), HIGH_Q)]
    result = run_analysis(
        fastq_bytes(records), F, R, params=AnalysisParams(r_convention="literal")
    )
    probe = result.run_summary["inputs"]["r_convention"]
    assert probe["requested"] == "literal"
    assert probe["reads_probed"] == 0


# --- derived parameters -----------------------------------------------------


def test_max_insert_length_defaults_to_three_times_the_longest_library_member():
    assert resolve_max_insert_length(AnalysisParams(), LIBRARY) == 60
    assert resolve_max_insert_length(AnalysisParams(), None) == 5000
    assert resolve_max_insert_length(AnalysisParams(max_insert_length=77), LIBRARY) == 77
    assert (
        resolve_max_insert_length(AnalysisParams(library_length_multiplier=10), LIBRARY)
        == 200
    )


def test_library_derived_cutoff_flags_an_oversized_insert():
    records = [
        ("normal", amplicon(LIB_A), HIGH_Q),
        ("oversized", amplicon("ACGT" * 40), HIGH_Q),  # 160 bp > 3 x 20
    ]
    result = run_analysis(
        fastq_bytes(records), F, R, library=LIBRARY,
        params=AnalysisParams(r_convention="literal"),
    )
    counts = result.run_summary["extraction"]["status_counts"]
    assert counts["insert_too_long"] == 1
    assert counts["extracted"] == 1
    assert result.run_summary["parameters"]["max_insert_length_used"] == 60


# --- inputs, outputs, serialisation ----------------------------------------


def test_path_and_bytes_inputs_give_identical_results(write_fastq, write_library):
    records = mixed_records()
    from_path = run_analysis(
        write_fastq(records), F, R, library=write_library(LIBRARY), params=base_params()
    )
    from_bytes = run_analysis(
        fastq_bytes(records), F, R, library=LIBRARY, params=base_params()
    )
    assert from_path.unique_sequences.equals(from_bytes.unique_sequences)
    assert (
        from_path.run_summary["extraction"]["status_counts"]
        == from_bytes.run_summary["extraction"]["status_counts"]
    )


def test_run_summary_is_json_serialisable_and_records_the_parameters():
    result = run_analysis(fastq_bytes(mixed_records()), F, R, params=base_params())
    text = json.dumps(result.run_summary)
    restored = json.loads(text)
    assert restored["parameters"]["min_mean_quality"] == 20.0
    assert restored["parameters"]["forward_max_edits_used"] == 2
    assert restored["parameters"]["max_insert_length_used"] == 200
    assert restored["inputs"]["forward_flank"] == F
    assert "timestamp_utc" in restored


def test_qc_summary_frame_is_flat_metric_value_rows():
    result = run_analysis(fastq_bytes(mixed_records()), F, R, params=base_params())
    frame = result.qc_summary
    assert list(frame.columns) == ["metric", "value"]
    metrics = set(frame["metric"])
    assert "quality.total_reads" in metrics
    assert "extraction.status_counts.extracted" in metrics


def test_write_outputs_creates_every_file(tmp_path):
    result = run_analysis(
        fastq_bytes(mixed_records()), F, R, library=LIBRARY, params=base_params()
    )
    written = result.write_outputs(
        tmp_path / "out", output_format="tsv", write_per_read=True, make_plots=True
    )
    for label in (
        "unique_sequences", "qc_summary", "library_comparison", "per_read",
        "run_summary_json", "run_summary_txt",
        "plot_quality_distribution", "plot_read_length_distribution",
        "plot_insert_length_distribution", "plot_top_sequences",
    ):
        assert label in written, label
        assert written[label].exists() and written[label].stat().st_size > 0

    header = written["unique_sequences"].read_text().splitlines()[0]
    assert "\t" in header and "," not in header


def test_empty_fastq_does_not_crash():
    result = run_analysis(b"", F, R, params=base_params())
    assert result.run_summary["quality"]["total_reads"] == 0
    assert result.unique_sequences.empty
    assert result.run_summary["quantification"]["unique_sequences"] == 0
    assert result.text_summary()


def test_invalid_flank_is_rejected_with_a_clear_message():
    import pytest

    with pytest.raises(ValueError, match="non-DNA"):
        run_analysis(fastq_bytes(mixed_records()), "ACGTXYZ", R)
    with pytest.raises(ValueError, match="empty"):
        run_analysis(fastq_bytes(mixed_records()), "", R)
