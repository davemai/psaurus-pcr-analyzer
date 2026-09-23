"""Regression tests: one per bug found in the audit of this pipeline.

Each test names the defect it pins down, so a future change that reintroduces
it fails with an explanation rather than a mystery.
"""

from __future__ import annotations

import gzip
import io
import tarfile

import pytest
from conftest import F, HIGH_Q, LIB_A, LIB_B, LIBRARY, R, amplicon, fastq_bytes, fastq_text

from psaurus_pcr.batch import run_batch
from psaurus_pcr.config import AnalysisParams
from psaurus_pcr.flanks import STATUS_EXTRACTED, extract_insert, find_flank
from psaurus_pcr.inputs import SampleInput, _uniquify, discover_inputs, list_sample_names
from psaurus_pcr.library import compare_to_library
from psaurus_pcr.pipeline import run_analysis
from psaurus_pcr.plots import top_sequences_figure
from psaurus_pcr.quantify import quantification_summary, quantify
from psaurus_pcr.sequtils import summarise_numeric, summarise_weighted


def params():
    return AnalysisParams(max_insert_length=200, r_convention="literal")


# --- 1: insert-length stats expanded one entry per read --------------------


def test_weighted_summary_matches_the_expanded_version():
    """summarise_weighted must agree exactly with expanding the counts.

    It replaced an O(reads) expansion that would have cost hundreds of MB on a
    multi-million-read flowcell, so it has to be numerically identical.
    """
    pairs = [(23, 3), (24, 400), (25, 17), (26, 1)]
    expanded = [value for value, count in pairs for _ in range(count)]
    assert summarise_weighted(pairs) == summarise_numeric(expanded)
    assert summarise_weighted([]) == summarise_numeric([])
    assert summarise_weighted([(42, 1)]) == summarise_numeric([42])
    # Zero-count entries must not shift the quantiles.
    assert summarise_weighted(pairs + [(999, 0)]) == summarise_numeric(expanded)


def test_quantification_summary_schema_is_the_same_when_empty():
    """A sample with nothing extracted must emit the same metric keys.

    Otherwise it contributes a ragged column to the combined QC table of a
    batch, where every sample is supposed to line up.
    """
    populated = quantification_summary(quantify([]))
    assert "insert_length_unique_sequences" in populated

    result = run_analysis(fastq_bytes([("r1", amplicon(LIB_A), HIGH_Q)]), F, R,
                          params=params())
    assert set(quantification_summary(result.unique_sequences)) == set(populated)


# --- 2: degenerate (IUPAC) flanks could never match ------------------------


def test_degenerate_flank_matches_without_spending_the_edit_budget():
    """An N in a flank must match any base, not count as a mismatch.

    validate_flank has always accepted IUPAC codes, so a degenerate primer was
    silently unmatchable: every ambiguous position ate an edit.
    """
    degenerate = F[:8] + "NNN" + F[11:]
    read = amplicon(LIB_A)
    hit = find_flank(degenerate, read, max_edits=0)
    assert hit is not None and hit.edit_distance == 0

    result = extract_insert("r", read, degenerate, R, AnalysisParams(
        forward_max_edits=0, reverse_max_edits=0), 200)
    assert result.status == STATUS_EXTRACTED
    assert result.insert == LIB_A


def test_ambiguity_code_flank_matches_its_constituent_bases():
    base = F[3]
    code = {"A": "R", "G": "R", "C": "Y", "T": "Y"}[base]  # R=AG, Y=CT
    degenerate = F[:3] + code + F[4:]
    hit = find_flank(degenerate, amplicon(LIB_A), max_edits=0)
    assert hit is not None and hit.edit_distance == 0


def test_an_n_in_the_read_does_not_break_a_clean_flank():
    read = amplicon(LIB_A)
    index = read.index(F) + 5
    with_n = read[:index] + "N" + read[index + 1:]
    hit = find_flank(F, with_n, max_edits=0)
    assert hit is not None and hit.edit_distance == 0


# --- 3: sample-name collisions ---------------------------------------------


def test_uniquify_handles_a_real_file_named_like_a_generated_suffix():
    """Two "sample" files plus a real "sample__2" must give three names.

    The old per-name counter produced "sample__2" twice, so one sample
    silently overwrote the other's output directory.
    """
    names = _uniquify(["sample", "sample", "sample__2"])
    assert len(set(names)) == 3, names
    assert names[0] == "sample"
    # The real "sample__2" keeps a name derived from its own filename rather
    # than being renumbered, so the mapping back to the file stays obvious.
    assert names[2].startswith("sample__2")
    assert len(set(_uniquify(["a", "a", "a__2", "a__2", "a"]))) == 5


def test_sample_named_like_a_matrix_column_does_not_break_the_batch(write_fastq_dir):
    """A file called length.fastq used to crash build_sequence_count_matrix.

    Sample names become *columns* of the cross-sample matrix, so one that
    collides with a metadata column ("length", "total_count", ...) has to be
    renamed at discovery.
    """
    records = [("r%d" % i, amplicon(LIB_A), HIGH_Q) for i in range(2)]
    root = write_fastq_dir({"length.fastq": records, "total_count.fastq": records,
                            "barcode01.fastq": records})
    batch = run_batch(root, F, R, params=params())
    assert batch.sample_names == ["barcode01", "length_sample", "total_count_sample"]
    matrix = batch.sequence_count_matrix
    assert int(matrix.loc[0, "total_count"]) == 6
    assert int(matrix.loc[0, "length"]) == len(LIB_A)
    assert int(matrix.loc[0, "length_sample"]) == 2


def test_ordinary_sample_names_are_left_alone(write_fastq_dir):
    """Only names that really would collide get renamed."""
    records = [("r1", amplicon(LIB_A), HIGH_Q)]
    root = write_fastq_dir({"sample.fastq": records, "count.fastq": records})
    assert [s.name for s in discover_inputs(root)] == ["count", "sample"]


# --- 4: batch output layout could overwrite itself -------------------------


def test_flat_layout_is_refused_for_a_multi_sample_batch(tmp_path):
    """The CLI guarded this; the Python API silently overwrote each sample."""
    records = [("r1", amplicon(LIB_A), HIGH_Q)]
    batch = run_batch(
        [SampleInput("s1", fastq_bytes(records), "s1"),
         SampleInput("s2", fastq_bytes(records), "s2")],
        F, R, params=params(),
    )
    with pytest.raises(ValueError, match="exactly one sample"):
        batch.write_outputs(tmp_path / "flat", sample_dirs=False)


# --- 5: archive detection ---------------------------------------------------


def test_gzipped_tar_without_a_tar_extension_is_still_expanded(tmp_path):
    """An upload called "order.gz" that is really a tarball must work.

    It used to be treated as a single gzipped FASTQ and then failed to parse.
    """
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
        for name in ("barcode01.fastq", "barcode02.fastq"):
            data = fastq_text([("r1", amplicon(LIB_A), HIGH_Q)]).encode("utf-8")
            info = tarfile.TarInfo(name)
            info.size = len(data)
            archive.addfile(info, io.BytesIO(data))
    path = tmp_path / "order.gz"
    path.write_bytes(buffer.getvalue())

    handle = io.BytesIO(path.read_bytes())
    handle.name = "order.gz"
    assert [s.name for s in discover_inputs(handle)] == ["barcode01", "barcode02"]


def test_a_real_gzipped_fastq_is_not_mistaken_for_an_archive():
    handle = io.BytesIO(gzip.compress(fastq_bytes([("r1", amplicon(LIB_A), HIGH_Q)])))
    handle.name = "barcode01.fastq.gz"
    samples = discover_inputs(handle)
    assert [s.name for s in samples] == ["barcode01"]


# --- 6: the web app's cheap preview must agree with the real discovery -----


def test_preview_and_discovery_agree(write_fastq_dir, write_zip):
    records = [("r1", amplicon(LIB_A), HIGH_Q)]
    root = write_fastq_dir({"barcode01.fastq": records, "nested/barcode02.fastq.gz": records})
    archive = write_zip({"order/barcode03.fastq": records, "order/notes.txt": []})

    preview = list_sample_names([root, archive])
    discovered = [(s.name, s.describe()) for s in discover_inputs([root, archive])]
    assert preview == discovered


def test_preview_names_an_upload_the_way_the_cli_does():
    """Uploads used to keep their full filename as the sample name."""
    handle = io.BytesIO(fastq_bytes([("r1", amplicon(LIB_A), HIGH_Q)]))
    handle.name = "barcode07.fastq.gz"
    assert list_sample_names(handle) == [("barcode07", "barcode07.fastq.gz")]


# --- 7: library comparison edge cases --------------------------------------


def test_library_comparison_on_an_empty_count_table():
    comparison = compare_to_library(quantify([]), LIBRARY)
    assert comparison.summary["library_sequences_detected"] == 0
    assert comparison.summary["percent_library_detected"] == 0.0
    assert comparison.summary["unexpected_unique_sequences"] == 0
    assert comparison.annotated_unique.empty
    assert len(comparison.library_table) == len(LIBRARY)


def test_library_file_with_only_comments_is_handled():
    result = run_analysis(
        fastq_bytes([("r1", amplicon(LIB_A), HIGH_Q)]), F, R,
        library=["# nothing here", ""], params=params(),
    )
    assert result.run_summary["library"]["library_sequences_unique"] == 0
    assert result.run_summary["library"]["percent_library_detected"] == 0.0
    assert result.run_summary["library"]["library_entries_rejected"] == 0


def test_fuzzy_matching_length_prefilter_does_not_change_the_answer():
    """The length shortcut is an optimisation; results must be identical."""
    one_off = LIB_A[:7] + ("G" if LIB_A[7] != "G" else "C") + LIB_A[8:]
    shorter = LIB_B[:-3]
    frame = quantify([])
    from psaurus_pcr.flanks import ExtractionResult

    extractions = [
        ExtractionResult("a", STATUS_EXTRACTED, insert=one_off, insert_length=len(one_off)),
        ExtractionResult("b", STATUS_EXTRACTED, insert=shorter, insert_length=len(shorter)),
    ]
    frame = quantify(extractions)
    fuzzy = compare_to_library(
        frame, LIBRARY, AnalysisParams(fuzzy_library=True, library_edit_fraction=0.05)
    )
    annotated = fuzzy.annotated_unique.set_index("sequence")
    assert annotated.loc[one_off, "library_match_type"] == "fuzzy"
    # 3 bases short of a 20 bp member is 3 edits, over the 1-edit budget.
    assert annotated.loc[shorter, "library_match_type"] == "none"


# --- 8: plotting guard ------------------------------------------------------


def test_top_sequences_figure_survives_top_n_of_zero():
    result = run_analysis(fastq_bytes([("r1", amplicon(LIB_A), HIGH_Q)]), F, R,
                          params=params())
    figure = top_sequences_figure(result.unique_sequences, top_n=0)
    assert figure is not None


def test_empty_fastq_file_on_disk_does_not_crash(tmp_path):
    empty = tmp_path / "empty.fastq"
    empty.write_text("")
    result = run_analysis(empty, F, R, params=params())
    assert result.run_summary["quality"]["total_reads"] == 0


def test_a_spreadsheet_header_does_not_become_a_library_member():
    """Rejected entries must not deflate "% of library detected"."""
    result = run_analysis(
        fastq_bytes([("r%d" % i, amplicon(LIB_A), HIGH_Q) for i in range(3)]),
        F, R, library=["sequence"] + LIBRARY, params=params(),
    )
    lib = result.run_summary["library"]
    assert lib["library_sequences_unique"] == len(LIBRARY)
    assert lib["library_entries_rejected"] == 1
    assert lib["library_entries_rejected_examples"] == ["sequence"]
    assert lib["library_sequences_detected"] == 1
    assert lib["percent_library_detected"] == pytest.approx(33.3333, abs=1e-3)


def test_a_rejected_entry_cannot_inflate_the_max_insert_length():
    """longest_library_sequence must ignore entries prepare_library rejects."""
    from psaurus_pcr.library import longest_library_sequence
    from psaurus_pcr.pipeline import resolve_max_insert_length

    assert longest_library_sequence(["ACGT", "a_very_long_header_row_from_excel"]) == 4
    assert resolve_max_insert_length(
        AnalysisParams(), ["ACGT", "a_very_long_header_row_from_excel"]
    ) == 12


def test_batch_plots_and_report_survive_a_batch_with_no_reads():
    """Every sample empty left no fate category to draw, and an empty legend
    made matplotlib raise from inside the PDF report."""
    from psaurus_pcr.plots import build_batch_figures
    from psaurus_pcr.report import report_bytes

    batch = run_batch(
        [SampleInput("a", b"", "a.fastq"), SampleInput("b", b"", "b.fastq")],
        F, R, params=params(),
    )
    assert batch.batch_summary["totals"]["total_reads"] == 0
    figures = build_batch_figures(batch)
    assert "batch_read_fate" in figures
    assert report_bytes(batch).startswith(b"%PDF-")


def test_batch_with_no_reads_still_writes_its_outputs(tmp_path):
    batch = run_batch([SampleInput("a", b"", "a.fastq"),
                       SampleInput("b", b"", "b.fastq")], F, R, params=params())
    written = batch.write_outputs(tmp_path / "empty_batch")
    assert written["sample_overview"].exists()
    assert written["batch_report_pdf"].exists()


def test_a_one_shot_stream_fails_loudly_instead_of_reporting_zero_reads():
    """Auto R-detection needs two passes; a pipe cannot give them.

    This used to report total_reads=0, indistinguishable from an empty file.
    """
    import io

    class OneShot(io.RawIOBase):
        def __init__(self, data):
            self._buffer = io.BytesIO(data)

        def read(self, size=-1):
            return self._buffer.read(size)

        def readable(self):
            return True

        def seekable(self):
            return False

        def seek(self, *args):
            raise OSError("not seekable")

    records = [("r%d" % i, amplicon(LIB_A), HIGH_Q) for i in range(4)]
    with pytest.raises(ValueError, match="could not be read a second time"):
        run_analysis(OneShot(fastq_bytes(records)), F, R)

    # With the convention pinned, one pass is enough and the same stream works.
    result = run_analysis(
        OneShot(fastq_bytes(records)), F, R,
        params=AnalysisParams(r_convention="literal", max_insert_length=200),
    )
    assert result.run_summary["quality"]["total_reads"] == 4
