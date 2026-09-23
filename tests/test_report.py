"""The PDF run report."""

from __future__ import annotations

import io

import pytest
from conftest import F, HIGH_Q, LIB_A, LIB_B, LIBRARY, LOW_Q, R, amplicon, fastq_bytes

from psaurus_pcr.batch import run_batch
from psaurus_pcr.config import AnalysisParams
from psaurus_pcr.inputs import SampleInput
from psaurus_pcr.pipeline import run_analysis
from psaurus_pcr.report import format_table, report_bytes, table_blocks, write_report

import pandas as pd

PDF_MAGIC = b"%PDF-"


def params():
    return AnalysisParams(max_insert_length=200, r_convention="literal")


def one_sample():
    records = [("r%d" % i, amplicon(LIB_A), HIGH_Q) for i in range(5)]
    records += [("b%d" % i, amplicon(LIB_B), HIGH_Q) for i in range(2)]
    records += [("bad", amplicon(LIB_A), LOW_Q)]
    return run_analysis(fastq_bytes(records), F, R, library=LIBRARY, params=params())


def a_batch():
    def sample(name, insert, n):
        return SampleInput(
            name, fastq_bytes([(f"{name}_{i}", amplicon(insert), HIGH_Q) for i in range(n)]),
            f"{name}.fastq",
        )
    return run_batch([sample("s1", LIB_A, 5), sample("s2", LIB_B, 3)],
                     F, R, library=LIBRARY, params=params())


def _text(data: bytes) -> str:
    pypdf = pytest.importorskip("pypdf", reason="pypdf is a test-only extra")
    reader = pypdf.PdfReader(io.BytesIO(data))
    return "\n".join(page.extract_text() or "" for page in reader.pages)


def _pages(data: bytes) -> int:
    pypdf = pytest.importorskip("pypdf", reason="pypdf is a test-only extra")
    return len(pypdf.PdfReader(io.BytesIO(data)).pages)


# --- single sample ----------------------------------------------------------


def test_sample_report_is_a_valid_pdf():
    data = report_bytes(one_sample())
    assert data.startswith(PDF_MAGIC)
    assert len(data) > 10_000
    assert _pages(data) >= 4


def test_sample_report_records_the_run():
    """The PDF must stand on its own: inputs, parameters and results."""
    result = one_sample()
    text = _text(report_bytes(result))
    assert "Plasmidsaurus PCR amplicon analysis" in text
    assert F in text and R in text
    assert "Minimum mean read quality" in text
    assert "Total reads" in text
    assert "Read fate" in text
    assert "Intended library comparison" in text
    assert result.run_summary["timestamp_local"] in text
    assert LIB_A in text


def test_sample_report_without_a_library_skips_that_section():
    result = run_analysis(
        fastq_bytes([("r1", amplicon(LIB_A), HIGH_Q)]), F, R, params=params()
    )
    text = _text(report_bytes(result))
    assert "Intended library comparison" not in text
    assert "(none)" in text


def test_report_accepts_a_path_and_a_stream(tmp_path):
    result = one_sample()
    path = tmp_path / "report.pdf"
    write_report(result, path)
    assert path.read_bytes().startswith(PDF_MAGIC)

    buffer = io.BytesIO()
    write_report(result, buffer)
    assert buffer.getvalue().startswith(PDF_MAGIC)


# --- batch ------------------------------------------------------------------


def test_batch_report_covers_the_overview_and_every_sample():
    batch = a_batch()
    text = _text(report_bytes(batch))
    assert "Batch report" in text
    assert "Batch overview" in text
    assert "Read fate counts per sample" in text
    for name in batch.sample_names:
        assert f"Sample: {name}" in text
    assert "Sequence x sample counts" in text
    assert "Library member x sample counts" in text


def test_batch_report_lists_failed_samples():
    broken = SampleInput("broken", b"@header_only\nACGT\n", "broken.fastq")
    good = SampleInput(
        "good", fastq_bytes([("r1", amplicon(LIB_A), HIGH_Q)]), "good.fastq"
    )
    batch = run_batch([good, broken], F, R, params=params())
    text = _text(report_bytes(batch))
    assert "Samples that failed" in text
    assert "broken" in text


def test_batch_report_caps_per_sample_detail():
    """A 96-barcode plate must not produce a 500-page document."""
    samples = [
        SampleInput(f"s{i:02d}", fastq_bytes([("r1", amplicon(LIB_A), HIGH_Q)]),
                    f"s{i:02d}.fastq")
        for i in range(6)
    ]
    batch = run_batch(samples, F, R, params=params())
    capped = _text(report_bytes(batch, max_detailed_samples=2))
    assert "Sample: s00" in capped
    assert "Sample: s05" not in capped
    assert "Detailed pages were produced for the first 2" in capped

    overview_only = report_bytes(batch, per_sample_detail=False)
    assert _pages(overview_only) < _pages(report_bytes(batch))


# --- output wiring ----------------------------------------------------------


def test_write_outputs_emits_report_pdf(tmp_path):
    written = one_sample().write_outputs(tmp_path / "out", make_plots=False)
    assert written["report_pdf"].exists()
    assert written["report_pdf"].read_bytes().startswith(PDF_MAGIC)


def test_write_outputs_can_skip_the_pdf(tmp_path):
    written = one_sample().write_outputs(tmp_path / "out", make_plots=False, make_pdf=False)
    assert "report_pdf" not in written
    assert not (tmp_path / "out" / "report.pdf").exists()


def test_batch_write_outputs_emits_one_combined_pdf(tmp_path):
    batch = a_batch()
    out = tmp_path / "batch"
    written = batch.write_outputs(out, make_plots=False)
    assert written["batch_report_pdf"].exists()
    # One report for the batch, not a duplicate inside every sample directory.
    for name in batch.sample_names:
        assert not (out / name / "report.pdf").exists()


# --- table rendering --------------------------------------------------------


def test_wide_tables_split_by_column_instead_of_dropping_columns():
    """A record-keeping report must never silently lose a column."""
    frame = pd.DataFrame({f"column_{i:02d}": [f"value{i}"] * 2 for i in range(30)})
    blocks = table_blocks(frame, max_rows=None, key_columns=1)
    assert len(blocks) > 1
    rendered = "\n".join(line for block in blocks for line in block)
    for column in frame.columns:
        assert column in rendered
    # The key column is repeated so every block can be read alone.
    assert all("column_00" in block[0] for block in blocks)


def test_truncated_tables_say_so():
    frame = pd.DataFrame({"sequence": [f"ACGT{i}" for i in range(100)],
                          "count": list(range(100))})
    lines = format_table(frame, max_rows=10)
    body = [line for line in lines if line.startswith("ACGT")]
    assert len(body) == 10
    assert any("showing 10 of 100 rows" in line for line in lines)


def test_empty_table_renders_a_placeholder():
    assert format_table(pd.DataFrame()) == ["(no rows)"]
