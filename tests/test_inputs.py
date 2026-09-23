"""Discovery of samples from files, folders, archives and buffers."""

from __future__ import annotations

import gzip
import io

import pytest
from conftest import HIGH_Q, LIB_A, LIB_B, amplicon, fastq_bytes

from psaurus_pcr.inputs import SampleInput, discover_inputs, looks_like_fastq, sample_name

A = [("a1", amplicon(LIB_A), HIGH_Q)]
B = [("b1", amplicon(LIB_B), HIGH_Q)]


@pytest.mark.parametrize(
    "filename,expected",
    [
        ("reads.fastq", True),
        ("reads.fq", True),
        ("reads.fastq.gz", True),
        ("reads.FQ.GZ", True),
        ("nested/dir/barcode07.fastq.gz", True),
        ("reads.txt", False),
        ("reads.fasta", False),
        ("summary.csv", False),
        (".hidden.fastq", False),
        ("", False),
    ],
)
def test_looks_like_fastq(filename, expected):
    assert looks_like_fastq(filename) is expected


@pytest.mark.parametrize(
    "filename,expected",
    [
        ("barcode07.fastq.gz", "barcode07"),
        ("barcode07.fq", "barcode07"),
        ("plate1/A01.fastq", "A01"),
        ("sample_1.FASTQ.GZ", "sample_1"),
        ("no_extension", "no_extension"),
    ],
)
def test_sample_name(filename, expected):
    assert sample_name(filename) == expected


def test_single_file_stays_a_path(write_fastq):
    path = write_fastq(A, name="barcode01.fastq")
    samples = discover_inputs(path)
    assert len(samples) == 1
    assert samples[0].name == "barcode01"
    # Kept as a path, not slurped, so large files still stream.
    assert samples[0].source == path


def test_several_files_are_separate_samples(write_fastq):
    first = write_fastq(A, name="s1.fastq")
    second = write_fastq(B, name="s2.fastq")
    samples = discover_inputs([first, second])
    assert [s.name for s in samples] == ["s1", "s2"]


def test_directory_is_searched_recursively_and_non_fastq_ignored(write_fastq_dir):
    root = write_fastq_dir({
        "barcode01.fastq": A,
        "barcode02.fastq.gz": B,
        "nested/barcode03.fq": A,
    })
    (root / "report.txt").write_text("not a fastq")
    (root / "nested" / "summary.csv").write_text("also not")

    samples = discover_inputs(root)
    assert [s.name for s in samples] == ["barcode01", "barcode02", "barcode03"]

    shallow = discover_inputs(root, recursive=False)
    assert [s.name for s in shallow] == ["barcode01", "barcode02"]


def test_directory_pattern_filters_the_search(write_fastq_dir):
    root = write_fastq_dir({
        "barcode01.fastq.gz": A,
        "barcode02.fastq.gz": B,
        "unclassified.fastq.gz": A,
    })
    samples = discover_inputs(root, pattern="barcode*.fastq.gz")
    assert [s.name for s in samples] == ["barcode01", "barcode02"]


def test_zip_members_are_read_into_memory(write_zip):
    archive = write_zip({
        "order/barcode01.fastq": A,
        "order/barcode02.fastq.gz": B,
        "order/notes.txt": [],
        "__MACOSX/._barcode01.fastq": A,
    })
    samples = discover_inputs(archive)
    assert [s.name for s in samples] == ["barcode01", "barcode02"]
    assert all(isinstance(s.source, bytes) for s in samples)
    assert samples[0].container == str(archive)
    assert "barcode01" in samples[0].describe()


def test_tar_gz_members_are_read(write_targz):
    archive = write_targz({"run/barcode01.fastq": A, "run/barcode02.fastq": B})
    samples = discover_inputs(archive)
    assert [s.name for s in samples] == ["barcode01", "barcode02"]
    assert all(isinstance(s.source, bytes) for s in samples)


def test_zip_bytes_are_detected_by_magic(write_zip):
    archive = write_zip({"x/barcode01.fastq": A})
    samples = discover_inputs(archive.read_bytes())
    assert [s.name for s in samples] == ["barcode01"]


def test_raw_fastq_bytes_and_gzip_bytes_are_one_sample():
    assert len(discover_inputs(fastq_bytes(A))) == 1
    assert len(discover_inputs(gzip.compress(fastq_bytes(A)))) == 1


def test_file_like_object_uses_its_name():
    """The Streamlit upload shape: a buffer with a .name attribute."""
    buffer = io.BytesIO(fastq_bytes(A))
    buffer.name = "barcode09.fastq"
    samples = discover_inputs(buffer)
    assert [s.name for s in samples] == ["barcode09"]
    assert samples[0].source == fastq_bytes(A)


def test_mixed_sources_are_combined_in_order(write_fastq, write_zip):
    single = write_fastq(A, name="loose.fastq")
    archive = write_zip({"z/barcode01.fastq": B})
    samples = discover_inputs([single, archive])
    assert [s.name for s in samples] == ["loose", "barcode01"]


def test_duplicate_sample_names_are_made_unique(write_fastq_dir):
    root = write_fastq_dir({
        "run1/sample.fastq": A,
        "run2/sample.fastq": B,
    })
    samples = discover_inputs(root)
    assert [s.name for s in samples] == ["sample", "sample__2"]
    # The origins still distinguish them.
    assert samples[0].origin != samples[1].origin


def test_sample_input_passes_through_unchanged():
    given = SampleInput(name="mine", source=fastq_bytes(A), origin="memory")
    assert discover_inputs(given) == [given]


def test_missing_path_raises():
    with pytest.raises(FileNotFoundError, match="input not found"):
        discover_inputs("/definitely/not/here.fastq")


def test_directory_with_no_fastq_returns_nothing(tmp_path):
    (tmp_path / "readme.md").write_text("nothing here")
    assert discover_inputs(tmp_path) == []


def test_unsupported_input_type_is_rejected():
    with pytest.raises(TypeError, match="Unsupported input"):
        discover_inputs([42])
