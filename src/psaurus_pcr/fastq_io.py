"""FASTQ / library-file input that works from paths *or* in-memory data.

Streamlit hands us an uploaded file object (bytes in memory); the CLI hands us
a path that may be gzipped and many GB.  Both must go through the same code,
so every reader here accepts a ``source`` that is one of:

* ``str`` / :class:`pathlib.Path` -- a file on disk, transparently gunzipped
  when the magic bytes say so (regardless of extension);
* ``bytes`` / ``bytearray`` -- raw file contents, likewise auto-gunzipped;
* a file-like object with ``.read()`` -- e.g. ``st.file_uploader`` output,
  ``io.StringIO``, an open handle.

Parsing itself is delegated to Biopython, which validates the 4-line record
structure and decodes the Sanger/Phred+33 quality string for us.
"""

from __future__ import annotations

import gzip
import io
import os
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, List, Sequence, Union

from Bio import SeqIO

from psaurus_pcr.sequtils import normalise_sequence

Source = Union[str, os.PathLike, bytes, bytearray, io.IOBase, object]

GZIP_MAGIC = b"\x1f\x8b"


@dataclass(frozen=True)
class FastqRead:
    """One nanopore read: identifier, sequence and decoded Phred scores."""

    id: str
    sequence: str
    qualities: Sequence[int]

    @property
    def length(self) -> int:
        return len(self.sequence)


def _rewind(obj) -> None:
    seek = getattr(obj, "seek", None)
    if callable(seek):
        try:
            seek(0)
        except (OSError, ValueError):  # non-seekable stream; nothing to do
            pass


@contextmanager
def open_text_stream(source: Source, encoding: str = "utf-8") -> Iterator[io.TextIOBase]:
    """Yield a text-mode handle for ``source``, gunzipping when needed.

    The caller never has to care whether it was given a path, bytes or an
    upload widget's buffer.
    """
    # --- path on disk: stream it, never slurp (FASTQ files get large) -------
    if isinstance(source, (str, os.PathLike)):
        path = Path(source)
        with open(path, "rb") as raw:
            is_gz = raw.read(2) == GZIP_MAGIC
        if is_gz:
            with gzip.open(path, "rt", encoding=encoding) as handle:
                yield handle
        else:
            with open(path, "rt", encoding=encoding) as handle:
                yield handle
        return

    # --- already-in-memory bytes -------------------------------------------
    if isinstance(source, (bytes, bytearray)):
        data = bytes(source)
        if data[:2] == GZIP_MAGIC:
            with gzip.open(io.BytesIO(data), "rt", encoding=encoding) as handle:
                yield handle
        else:
            yield io.StringIO(data.decode(encoding))
        return

    # --- file-like ----------------------------------------------------------
    if hasattr(source, "read"):
        _rewind(source)
        chunk = source.read()
        if isinstance(chunk, str):
            yield io.StringIO(chunk)
            return
        data = bytes(chunk)
        if data[:2] == GZIP_MAGIC:
            with gzip.open(io.BytesIO(data), "rt", encoding=encoding) as handle:
                yield handle
        else:
            yield io.StringIO(data.decode(encoding))
        return

    raise TypeError(
        "Unsupported input source: expected a path, bytes, or a file-like "
        f"object with .read(), got {type(source).__name__}"
    )


def iter_fastq(source: Source) -> Iterator[FastqRead]:
    """Stream :class:`FastqRead` records from ``source``.

    Sequences are upper-cased on the way in so that downstream exact-match
    collapsing is not defeated by soft-masked or lowercase basecalls.
    """
    with open_text_stream(source) as handle:
        for record in SeqIO.parse(handle, "fastq"):
            yield FastqRead(
                id=record.id,
                sequence=str(record.seq).upper(),
                # Biopython decodes Phred+33 for us and validates the range.
                qualities=record.letter_annotations["phred_quality"],
            )


def read_fastq(source: Source) -> List[FastqRead]:
    """Eagerly read all records (convenient for tests and small uploads)."""
    return list(iter_fastq(source))


def read_library(source: Source) -> List[str]:
    """Read an intended-library text file: one sequence per line.

    Blank lines and ``#`` comment lines are ignored.  FASTA-style ``>`` headers
    are also ignored, so a FASTA file of library members works as long as each
    sequence is on a single line.  Every entry is normalised (whitespace
    stripped, upper-cased) because case/whitespace differences between a design
    spreadsheet and a basecall are formatting, not biology.
    """
    sequences: List[str] = []
    with open_text_stream(source) as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped or stripped.startswith("#") or stripped.startswith(">"):
                continue
            sequences.append(normalise_sequence(stripped))
    return sequences
