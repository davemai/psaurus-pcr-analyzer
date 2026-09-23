"""Steps 3 and 4: collapsing/counting and intended-library comparison."""

from __future__ import annotations

from conftest import LIB_A, LIB_B, LIB_C, LIBRARY

from psaurus_pcr.config import AnalysisParams
from psaurus_pcr.flanks import STATUS_EXTRACTED, STATUS_TOO_LONG, ExtractionResult
from psaurus_pcr.library import compare_to_library, prepare_library
from psaurus_pcr.quantify import quantification_summary, quantify


def extracted(insert, read_id="r", status=STATUS_EXTRACTED):
    return ExtractionResult(read_id=read_id, status=status, insert=insert,
                            insert_length=len(insert), orientation="forward")


def test_counts_are_sorted_descending_with_percentages():
    results = [extracted(LIB_A) for _ in range(5)]
    results += [extracted(LIB_B) for _ in range(3)]
    results += [extracted(LIB_C)]
    frame = quantify(results)

    assert list(frame["sequence"]) == [LIB_A, LIB_B, LIB_C]
    assert list(frame["count"]) == [5, 3, 1]
    assert frame.loc[0, "percent_of_extracted"] == 55.555556
    assert list(frame["length"]) == [20, 20, 20]


def test_flagged_reads_do_not_enter_the_count_table():
    results = [extracted(LIB_A), extracted("A" * 9999, status=STATUS_TOO_LONG)]
    frame = quantify(results)
    assert list(frame["sequence"]) == [LIB_A]
    assert int(frame["count"].sum()) == 1


def test_quantification_summary_reports_singletons_and_lengths():
    results = [extracted(LIB_A) for _ in range(4)] + [extracted(LIB_B)]
    summary = quantification_summary(quantify(results))
    assert summary["total_extracted_reads"] == 5
    assert summary["unique_sequences"] == 2
    assert summary["singletons"] == 1
    assert summary["percent_singletons"] == 50.0
    assert summary["insert_length"]["median"] == 20


def test_empty_input_produces_an_empty_but_well_formed_table():
    frame = quantify([])
    assert frame.empty
    assert list(frame.columns) == ["sequence", "count", "percent_of_extracted", "length"]
    assert quantification_summary(frame)["unique_sequences"] == 0


# --- library comparison -----------------------------------------------------


def test_prepare_library_normalises_and_deduplicates():
    ids, seqs, duplicates, rejected = prepare_library([" acgt ", "ACGT", "", "TTTT"])
    assert seqs == ["ACGT", "TTTT"]
    assert duplicates == 1
    assert rejected == []
    assert ids == ["lib_0001", "lib_0004"]


def test_prepare_library_rejects_entries_that_are_not_dna():
    """A spreadsheet header or a comment must not become a library member."""
    ids, seqs, duplicates, rejected = prepare_library(
        ["sequence", "# a comment", ">header", "ACGT", "BARCODE_1", "TTTT"]
    )
    assert seqs == ["ACGT", "TTTT"]
    assert rejected == ["sequence", "BARCODE_1"]
    assert ids == ["lib_0004", "lib_0006"]


def test_exact_comparison_reports_coverage_and_unexpected_sequences():
    unexpected = "GGGGGGGGGGCCCCCCCCCC"
    results = [extracted(LIB_A) for _ in range(6)]
    results += [extracted(LIB_B) for _ in range(3)]
    results += [extracted(unexpected)]
    comparison = compare_to_library(quantify(results), LIBRARY)

    summary = comparison.summary
    assert summary["matching_mode"] == "exact"
    assert summary["library_sequences_unique"] == 3
    assert summary["library_sequences_detected"] == 2
    assert summary["percent_library_detected"] == 66.6667
    assert summary["unexpected_unique_sequences"] == 1
    assert summary["reads_in_unexpected_sequences"] == 1
    assert summary["percent_reads_unexpected"] == 10.0

    table = comparison.library_table.set_index("library_id")
    assert bool(table.loc["lib_0001", "detected"]) is True
    assert int(table.loc["lib_0001", "count"]) == 6
    assert bool(table.loc["lib_0003", "detected"]) is False
    assert int(table.loc["lib_0003", "count"]) == 0

    annotated = comparison.annotated_unique.set_index("sequence")
    assert annotated.loc[LIB_A, "library_match_type"] == "exact"
    assert annotated.loc[unexpected, "library_match_type"] == "none"
    assert bool(annotated.loc[unexpected, "in_library"]) is False


def test_comparison_is_robust_to_case_and_whitespace_in_the_library_file():
    results = [extracted(LIB_A)]
    messy = [f"  {LIB_A.lower()}  ", LIB_B]
    comparison = compare_to_library(quantify(results), messy)
    assert comparison.summary["library_sequences_detected"] == 1
    assert comparison.summary["unexpected_unique_sequences"] == 0


def test_near_miss_is_unexpected_under_exact_matching_but_matches_when_fuzzy():
    one_off = LIB_A[:7] + ("G" if LIB_A[7] != "G" else "C") + LIB_A[8:]
    frame = quantify([extracted(one_off) for _ in range(4)])

    strict = compare_to_library(frame, LIBRARY, AnalysisParams())
    assert strict.summary["library_sequences_detected"] == 0
    assert strict.summary["unexpected_unique_sequences"] == 1

    fuzzy = compare_to_library(
        frame, LIBRARY, AnalysisParams(fuzzy_library=True, library_edit_fraction=0.05)
    )
    assert fuzzy.summary["matching_mode"] == "fuzzy"
    assert fuzzy.summary["library_sequences_detected"] == 1
    assert fuzzy.summary["unexpected_unique_sequences"] == 0
    annotated = fuzzy.annotated_unique.set_index("sequence")
    assert annotated.loc[one_off, "library_match_type"] == "fuzzy"
    assert int(annotated.loc[one_off, "library_edit_distance"]) == 1


def test_fuzzy_matching_still_rejects_a_sequence_beyond_the_budget():
    far = LIB_A[:5] + "TTTTTTTTTT" + LIB_A[15:]  # ~10 edits from LIB_A
    frame = quantify([extracted(far)])
    fuzzy = compare_to_library(frame, LIBRARY, AnalysisParams(fuzzy_library=True))
    assert fuzzy.summary["unexpected_unique_sequences"] == 1


def test_fuzzy_matching_does_not_merge_a_truncated_product():
    """Global alignment means a half-length product is not 'a library member'."""
    truncated = LIB_A[:10]
    frame = quantify([extracted(truncated)])
    fuzzy = compare_to_library(frame, LIBRARY, AnalysisParams(fuzzy_library=True))
    assert fuzzy.summary["library_sequences_detected"] == 0
