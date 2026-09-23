"""Shared fixtures: synthetic nanopore-like FASTQ data built in memory.

Everything here is deterministic -- no randomness, no external files -- so a
failure always points at the code rather than at the test data.
"""

from __future__ import annotations

import gzip
import io
import sys
import tarfile
import zipfile
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple, Union

import pytest

# Allow ``pytest`` from a bare checkout (no install step required).
_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from psaurus_pcr.sequtils import reverse_complement  # noqa: E402

# --- building blocks -------------------------------------------------------

# Non-repetitive, mutually dissimilar 20 bp flanks: the closest pair is 8 edits
# apart and neither is similar to its own reverse complement, so a hit at the
# default 2-edit budget is never accidental.
F = "CAGTTCGGACTTAGCCATGA"          # 20 bp forward flank
R = "TGGACCAATCGTTACGGTCA"          # 20 bp reverse flank, on the F strand
R_AS_PRIMER = reverse_complement(R)  # how a wet-lab reverse primer would be written

PAD_LEFT = "CCATTGAGCTTACGGATCCAAT"
PAD_RIGHT = "AGGTTCCATAGCTTGACCAATG"

# Three "designed library" members, all 20 bp, mutually dissimilar.
LIB_A = "ACGTACGTACGTACGTAAAA"
LIB_B = "TTTTGGGGCCCCAAAATTTT"
LIB_C = "CAGTCAGTCAGTCAGTCAGT"
LIBRARY = [LIB_A, LIB_B, LIB_C]

HIGH_Q = 30   # comfortably above the Q20 default
LOW_Q = 5     # the sort of read Q20 filtering exists to remove

QualitySpec = Union[int, Sequence[int]]
Record = Tuple[str, str, QualitySpec]


def amplicon(insert: str, forward: str = F, reverse: str = R) -> str:
    """A full read: genomic padding, F, the insert, R, more padding."""
    return PAD_LEFT + forward + insert + reverse + PAD_RIGHT


def mutate(sequence: str, positions: Iterable[int]) -> str:
    """Substitute the base at each 0-based position with a different one.

    Used to simulate the residual basecall errors that make exact flank
    matching unworkable on nanopore data.
    """
    swap = {"A": "C", "C": "A", "G": "T", "T": "G", "N": "A"}
    chars = list(sequence)
    for pos in positions:
        chars[pos] = swap[chars[pos]]
    return "".join(chars)


def fastq_text(records: Sequence[Record]) -> str:
    """Render ``(id, sequence, quality)`` triples as Phred+33 FASTQ text."""
    blocks: List[str] = []
    for read_id, sequence, quality in records:
        if isinstance(quality, int):
            scores = [quality] * len(sequence)
        else:
            scores = list(quality)
        assert len(scores) == len(sequence), f"quality length mismatch for {read_id}"
        qual_string = "".join(chr(score + 33) for score in scores)
        blocks.append(f"@{read_id}\n{sequence}\n+\n{qual_string}")
    return "\n".join(blocks) + "\n"


def fastq_bytes(records: Sequence[Record]) -> bytes:
    """The same, as bytes -- mimics a Streamlit upload buffer."""
    return fastq_text(records).encode("utf-8")


@pytest.fixture
def write_fastq(tmp_path):
    """Factory writing records to a temporary .fastq file and returning the path."""

    def _write(records: Sequence[Record], name: str = "reads.fastq") -> Path:
        path = tmp_path / name
        path.write_text(fastq_text(records))
        return path

    return _write


@pytest.fixture
def write_library(tmp_path):
    """Factory writing a one-sequence-per-line library file."""

    def _write(sequences: Sequence[str], name: str = "library.txt") -> Path:
        path = tmp_path / name
        path.write_text("\n".join(sequences) + "\n")
        return path

    return _write


@pytest.fixture
def write_fastq_dir(tmp_path):
    """Factory writing ``{filename: records}`` into a directory tree.

    Keys may contain ``/`` to create nested folders, which is what a real
    Plasmidsaurus delivery or MinKNOW output looks like.  A ``.gz`` key is
    written gzipped.
    """

    def _write(mapping, dirname: str = "run") -> Path:
        root = tmp_path / dirname
        for relative, records in mapping.items():
            path = root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            text = fastq_text(records)
            if relative.endswith(".gz"):
                path.write_bytes(gzip.compress(text.encode("utf-8")))
            else:
                path.write_text(text)
        return root

    return _write


@pytest.fixture
def write_zip(tmp_path):
    """Factory writing ``{member_name: records}`` into a .zip archive."""

    def _write(mapping, name: str = "archive.zip") -> Path:
        path = tmp_path / name
        with zipfile.ZipFile(path, "w") as archive:
            for member, records in mapping.items():
                data = fastq_text(records).encode("utf-8")
                if member.endswith(".gz"):
                    data = gzip.compress(data)
                archive.writestr(member, data)
        return path

    return _write


@pytest.fixture
def write_targz(tmp_path):
    """Factory writing ``{member_name: records}`` into a .tar.gz archive."""

    def _write(mapping, name: str = "archive.tar.gz") -> Path:
        path = tmp_path / name
        with tarfile.open(path, "w:gz") as archive:
            for member, records in mapping.items():
                data = fastq_text(records).encode("utf-8")
                if member.endswith(".gz"):
                    data = gzip.compress(data)
                info = tarfile.TarInfo(member)
                info.size = len(data)
                archive.addfile(info, io.BytesIO(data))
        return path

    return _write
