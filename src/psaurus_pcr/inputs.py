"""Discovery of FASTQ samples from heterogeneous inputs.

A run rarely arrives as one tidy file.  Plasmidsaurus delivers a folder per
order; people zip that folder to email it; a cluster user points at a
directory of barcodes.  This module turns any of those into a flat, ordered
list of :class:`SampleInput` records that the batch runner can iterate over.

Accepted sources (individually, or mixed in one list):

* a FASTQ file path, plain or gzipped;
* a **directory** -- searched (recursively by default) for FASTQ files;
* a **.zip** archive -- FASTQ members are read out in memory;
* a **.tar / .tar.gz / .tgz** archive -- likewise;
* raw ``bytes`` of any of the above (a zip is detected by its magic bytes);
* a file-like object with ``.read()`` and usually ``.name`` -- i.e. exactly
  what ``st.file_uploader(accept_multiple_files=True)`` hands back.

Nothing is ever extracted to disk: archive members are read into memory and
handed to the pipeline as bytes, which is the same path a Streamlit upload
takes.
"""

from __future__ import annotations

import io
import os
import re
import tarfile
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Iterable, List, Optional, Sequence, Tuple, Union

from psaurus_pcr.fastq_io import GZIP_MAGIC, Source

ZIP_MAGIC = b"PK\x03\x04"

#: Extensions we treat as FASTQ, optionally followed by a compression suffix.
FASTQ_EXTENSIONS = (".fastq", ".fq")
COMPRESSION_EXTENSIONS = (".gz", ".bz2", ".xz")
TAR_EXTENSIONS = (".tar", ".tar.gz", ".tgz", ".tar.bz2", ".tbz", ".tar.xz", ".txz")

_SAMPLE_SUFFIX_RE = re.compile(
    r"(?:\.(?:fastq|fq))?(?:\.(?:gz|bz2|xz))?$", re.IGNORECASE
)


@dataclass(frozen=True)
class SampleInput:
    """One FASTQ file to analyse, wherever it came from.

    ``source`` is whatever :func:`psaurus_pcr.fastq_io.iter_fastq` needs: a
    path for files on disk (so large files stream rather than being slurped),
    or bytes for archive members and uploads.
    """

    name: str
    source: Source
    origin: str
    container: Optional[str] = None

    def describe(self) -> str:
        return self.origin if self.container is None else f"{self.container}!{self.origin}"


def looks_like_fastq(filename: str) -> bool:
    """True when ``filename`` has a FASTQ extension, with or without compression."""
    lowered = PurePosixPath(filename).name.lower()
    if not lowered or lowered.startswith("."):
        return False
    for compression in ("",) + COMPRESSION_EXTENSIONS:
        for extension in FASTQ_EXTENSIONS:
            if lowered.endswith(extension + compression):
                return True
    return False


def is_tar_name(filename: str) -> bool:
    lowered = PurePosixPath(filename).name.lower()
    return any(lowered.endswith(extension) for extension in TAR_EXTENSIONS)


def sample_name(filename: str) -> str:
    """Derive a sample label from a file name.

    ``barcode07.fastq.gz`` -> ``barcode07``; ``plate1/A01.fq`` -> ``A01``.
    Falls back to the whole name if nothing recognisable is stripped.
    """
    base = PurePosixPath(str(filename).replace("\\", "/")).name
    stripped = _SAMPLE_SUFFIX_RE.sub("", base)
    return stripped or base


def _is_hidden_member(member_name: str) -> bool:
    """Skip dotfiles and the junk macOS puts in zips."""
    parts = PurePosixPath(member_name).parts
    return any(part.startswith(".") for part in parts) or "__MACOSX" in parts


#: Names a sample may not take, because :mod:`psaurus_pcr.batch` turns sample
#: names into *columns* of the cross-sample matrices alongside these metadata
#: columns.  (Long-format columns such as ``sample`` or ``count`` are not
#: listed: there, sample names are cell values, not column headers, so
#: ``sample.fastq`` -- a perfectly ordinary filename -- is left alone.)
RESERVED_SAMPLE_NAMES = frozenset({
    "sequence", "length", "total_count", "n_samples_detected",
    "library_id", "library_sequence", "library_match_type", "in_library",
    "metric",
})


def _uniquify(names: Sequence[str]) -> List[str]:
    """Make sample names unique and safe, preserving order.

    Two barcodes called ``sample.fastq`` in different folders are different
    samples; silently overwriting one with the other would be much worse than
    calling the second one ``sample__2``.  The assigned names are checked
    against *all* names already handed out -- not just a per-name counter --
    so a real file called ``sample__2.fastq`` alongside two ``sample.fastq``
    still produces three distinct labels.  Names that would shadow a
    cross-sample table column are suffixed too.
    """
    taken: set = set()
    out: List[str] = []
    for name in names:
        candidate = f"{name}_sample" if name in RESERVED_SAMPLE_NAMES else name
        if candidate in taken:
            index = 2
            while f"{candidate}__{index}" in taken:
                index += 1
            candidate = f"{candidate}__{index}"
        taken.add(candidate)
        out.append(candidate)
    return out


# ---------------------------------------------------------------------------
# expanders
# ---------------------------------------------------------------------------


def _from_zip_bytes(data: bytes, container: str) -> List[SampleInput]:
    found: List[SampleInput] = []
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        for info in sorted(archive.infolist(), key=lambda i: i.filename):
            if info.is_dir() or _is_hidden_member(info.filename):
                continue
            if not looks_like_fastq(info.filename):
                continue
            found.append(
                SampleInput(
                    name=sample_name(info.filename),
                    source=archive.read(info),
                    origin=info.filename,
                    container=container,
                )
            )
    return found


def _from_tar_bytes(data: bytes, container: str) -> List[SampleInput]:
    found: List[SampleInput] = []
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:*") as archive:
        members = sorted(
            (m for m in archive.getmembers() if m.isfile()), key=lambda m: m.name
        )
        for member in members:
            if _is_hidden_member(member.name) or not looks_like_fastq(member.name):
                continue
            handle = archive.extractfile(member)
            if handle is None:  # pragma: no cover - defensive
                continue
            found.append(
                SampleInput(
                    name=sample_name(member.name),
                    source=handle.read(),
                    origin=member.name,
                    container=container,
                )
            )
    return found


def _from_directory(path: Path, recursive: bool, pattern: Optional[str]) -> List[SampleInput]:
    globber = path.rglob if recursive else path.glob
    candidates = sorted(
        entry for entry in globber(pattern or "*")
        if entry.is_file() and looks_like_fastq(entry.name)
    )
    return [
        SampleInput(name=sample_name(entry.name), source=entry, origin=str(entry))
        for entry in candidates
    ]


def _from_path(path: Path, recursive: bool, pattern: Optional[str]) -> List[SampleInput]:
    if path.is_dir():
        return _from_directory(path, recursive, pattern)
    if not path.exists():
        raise FileNotFoundError(f"input not found: {path}")
    if zipfile.is_zipfile(path):
        return _from_zip_bytes(path.read_bytes(), container=str(path))
    if is_tar_name(path.name) and tarfile.is_tarfile(path):
        return _from_tar_bytes(path.read_bytes(), container=str(path))
    # A plain FASTQ: keep it as a path so it streams instead of being slurped.
    return [SampleInput(name=sample_name(path.name), source=path, origin=str(path))]


def _from_bytes(data: bytes, label: str) -> List[SampleInput]:
    if data[:4] == ZIP_MAGIC:
        return _from_zip_bytes(data, container=label)
    if is_tar_name(label) or _looks_like_uncompressed_tar(data):
        try:
            return _from_tar_bytes(data, container=label)
        except tarfile.TarError:
            pass
    # A gzip blob that is not obviously a FASTQ may still be a tarball whose
    # name lost its extension (a browser upload called "order.gz", say), so
    # probe it before assuming one gzipped FASTQ.
    if data[:2] == GZIP_MAGIC and not looks_like_fastq(label):
        try:
            found = _from_tar_bytes(data, container=label)
        except tarfile.TarError:
            found = []
        if found:
            return found
    return [SampleInput(name=sample_name(label), source=data, origin=label)]


def _looks_like_uncompressed_tar(data: bytes) -> bool:
    # POSIX tar stores the magic "ustar" at offset 257.
    return len(data) > 262 and data[257:262] == b"ustar"


# ---------------------------------------------------------------------------
# the entry point
# ---------------------------------------------------------------------------


def discover_inputs(
    sources: Union[Source, "SampleInput", Iterable],
    recursive: bool = True,
    pattern: Optional[str] = None,
) -> List[SampleInput]:
    """Expand any mix of files, folders, archives and buffers into samples.

    Parameters
    ----------
    sources:
        One source or an iterable of them (see the module docstring).
    recursive:
        Search directories recursively.  Plasmidsaurus deliveries and MinKNOW
        output are both nested, so this defaults to True.
    pattern:
        Optional glob applied when walking a directory, e.g. ``"barcode*.fastq.gz"``.
        Ignored for archives and individual files.

    Returns
    -------
    list of :class:`SampleInput`, in a stable order, with unique ``name`` values.

    Raises
    ------
    FileNotFoundError
        If a named path does not exist.
    """
    if isinstance(sources, (str, os.PathLike, bytes, bytearray, SampleInput)) or hasattr(
        sources, "read"
    ):
        sources = [sources]

    collected: List[SampleInput] = []
    for source in sources:
        if isinstance(source, SampleInput):
            collected.append(source)
        elif isinstance(source, (str, os.PathLike)):
            collected.extend(_from_path(Path(source), recursive, pattern))
        elif isinstance(source, (bytes, bytearray)):
            collected.extend(_from_bytes(bytes(source), label="<in-memory>"))
        elif hasattr(source, "read"):
            label = getattr(source, "name", None) or "<in-memory>"
            getvalue = getattr(source, "getvalue", None)
            data = getvalue() if callable(getvalue) else source.read()
            if isinstance(data, str):
                data = data.encode("utf-8")
            collected.extend(_from_bytes(bytes(data), label=str(label)))
        else:
            raise TypeError(
                "Unsupported input: expected a path, directory, archive, bytes "
                f"or file-like object, got {type(source).__name__}"
            )

    # Re-label duplicates only after everything has been collected, so the
    # suffixes reflect the whole batch rather than one source at a time.
    unique = _uniquify([sample.name for sample in collected])
    return [
        SampleInput(name=new_name, source=s.source, origin=s.origin, container=s.container)
        for new_name, s in zip(unique, collected)
    ]


def list_sample_names(
    sources: Union[Source, "SampleInput", Iterable],
    recursive: bool = True,
    pattern: Optional[str] = None,
) -> List[Tuple[str, str]]:
    """Cheap preview: ``(sample_name, description)`` without reading the data.

    Used by the web app, which re-runs its whole script on every widget
    interaction -- fully expanding a multi-GB upload each time a slider moves
    would be unusable.  A zip is listed from its central directory (no member
    is decompressed) and a plain file from its name alone.  A tar cannot be
    listed without decompressing it, so that case falls back to
    :func:`discover_inputs`.

    The filtering predicates are shared with :func:`discover_inputs`, so the
    two always agree on which files count as FASTQ.
    """
    if isinstance(sources, (str, os.PathLike, bytes, bytearray, SampleInput)) or hasattr(
        sources, "read"
    ):
        sources = [sources]

    names: List[str] = []
    described: List[str] = []

    def add(name: str, description: str) -> None:
        names.append(name)
        described.append(description)

    def add_zip(data: bytes, label: str) -> bool:
        if data[:4] != ZIP_MAGIC:
            return False
        with zipfile.ZipFile(io.BytesIO(data)) as archive:
            for member in sorted(archive.namelist()):
                if member.endswith("/") or _is_hidden_member(member):
                    continue
                if looks_like_fastq(member):
                    add(sample_name(member), f"{label}!{member}")
        return True

    for source in sources:
        if isinstance(source, (str, os.PathLike)):
            path = Path(source)
            if path.is_file() and zipfile.is_zipfile(path):
                add_zip(path.read_bytes(), str(path))
                continue
            if path.is_file() and not is_tar_name(path.name):
                add(sample_name(path.name), str(path))
                continue
            for sample in _from_path(path, recursive, pattern):
                add(sample.name, sample.describe())
            continue

        data = None
        label = "<in-memory>"
        if isinstance(source, (bytes, bytearray)):
            data = bytes(source)
        elif hasattr(source, "read"):
            label = str(getattr(source, "name", None) or label)
            getvalue = getattr(source, "getvalue", None)
            raw = getvalue() if callable(getvalue) else source.read()
            data = raw.encode("utf-8") if isinstance(raw, str) else bytes(raw)
        if data is not None:
            if add_zip(data, label):
                continue
            if not is_tar_name(label) and not _looks_like_uncompressed_tar(data):
                add(sample_name(label), label)
                continue
        for sample in discover_inputs(source, recursive=recursive, pattern=pattern):
            add(sample.name, sample.describe())

    return list(zip(_uniquify(names), described))


class PathNotAllowed(ValueError):
    """A server-side path pointed outside the directory the app may read."""


def resolve_under_root(candidate: Union[str, os.PathLike], root: Optional[Union[str, os.PathLike]]) -> Path:
    """Resolve ``candidate`` and require it to sit inside ``root``.

    Any web front end that lets a visitor type a server-side path is handing
    out a file-read primitive unless the path is confined.  This does the
    confinement in one place so the rule is testable and cannot drift:

    * both sides are fully resolved first, so ``..`` segments and symlinks
      that escape the root are rejected rather than followed;
    * ``root=None`` means "no confinement", which is only for a host where the
      operator has explicitly opted in.

    Raises :class:`PathNotAllowed` if the path escapes, and
    :class:`FileNotFoundError` if it does not exist.
    """
    path = Path(candidate).expanduser()
    if root is None:
        resolved = path.resolve()
        if not resolved.exists():
            raise FileNotFoundError(f"input not found: {candidate}")
        return resolved

    root_path = Path(root).expanduser().resolve()
    # Resolve relative input against the root rather than the process's cwd,
    # so "examples/batch" means the same thing wherever the app is started.
    resolved = (path if path.is_absolute() else root_path / path).resolve()
    if not resolved.is_relative_to(root_path):
        raise PathNotAllowed(
            f"{candidate!r} is outside the directory this app is allowed to "
            f"read ({root_path})."
        )
    if not resolved.exists():
        raise FileNotFoundError(f"input not found: {candidate}")
    return resolved
