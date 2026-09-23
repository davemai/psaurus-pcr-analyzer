"""Input handling: paths, bytes, gzip and file-like objects must all work."""

from __future__ import annotations

import gzip
import io

from conftest import HIGH_Q, LIB_A, amplicon, fastq_bytes, fastq_text

from psaurus_pcr.fastq_io import read_fastq, read_library

RECORDS = [("r1", amplicon(LIB_A), HIGH_Q)]


def test_read_from_path(write_fastq):
    reads = read_fastq(write_fastq(RECORDS))
    assert len(reads) == 1
    assert reads[0].id == "r1"
    assert reads[0].sequence == amplicon(LIB_A)
    assert set(reads[0].qualities) == {HIGH_Q}


def test_read_from_bytes_and_stream():
    """The Streamlit path: bytes in memory, never touching disk."""
    from_bytes = read_fastq(fastq_bytes(RECORDS))
    from_stream = read_fastq(io.BytesIO(fastq_bytes(RECORDS)))
    from_text = read_fastq(io.StringIO(fastq_text(RECORDS)))
    assert from_bytes == from_stream == from_text


def test_gzip_is_detected_by_magic_bytes_not_extension(tmp_path):
    path = tmp_path / "reads.fastq"  # deliberately not .gz
    path.write_bytes(gzip.compress(fastq_bytes(RECORDS)))
    assert read_fastq(path) == read_fastq(fastq_bytes(RECORDS))
    assert read_fastq(gzip.compress(fastq_bytes(RECORDS))) == read_fastq(fastq_bytes(RECORDS))


def test_library_parsing_ignores_comments_headers_and_normalises_case(tmp_path):
    path = tmp_path / "library.txt"
    path.write_text(
        "# a comment\n"
        ">fasta_header\n"
        "  acgt acgt  \n"
        "\n"
        "TTTT\n"
    )
    assert read_library(path) == ["ACGTACGT", "TTTT"]
