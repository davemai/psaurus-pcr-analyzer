#!/usr/bin/env python3
"""Generate the small synthetic dataset shipped in ``examples/``.

Two datasets are produced:

* ``example_reads.fastq`` + ``example_library.txt`` -- a single sample, for the
  quickstart;
* ``batch/barcode0{1..4}.fastq.gz`` and ``example_batch.zip`` -- four samples
  with deliberately different behaviour (a clean one, a shallow one, a
  contaminated one, one with a library dropout), for the batch mode.

The reads imitate what a Plasmidsaurus amplicon run looks like: a defined
library of short barcodes between two constant flanks, sequenced in both
orientations, with a realistic per-base error rate, a low-quality tail of
reads, and a sprinkling of the artefacts the pipeline is meant to flag
(concatemers, missing flanks, empty and oversized inserts, and a sequence that
is not in the library at all).

Deterministic: a fixed seed, so the checked-in files can be regenerated
byte-for-byte with ``python examples/make_example_data.py``.
"""

from __future__ import annotations

import gzip
import io
import random
import zipfile
from pathlib import Path

SEED = 20260921
HERE = Path(__file__).resolve().parent

FORWARD_FLANK = "CAGTTCGGACTTAGCCATGACT"   # 22 bp
REVERSE_FLANK = "TGGACCAATCGTTACGGTCAAG"   # 22 bp, on the same strand as F

# An 8-member "intended library" of 24 bp barcodes.
LIBRARY = [
    "ACGTTGCAAGGTCCTTAAGGCATT",
    "TTCAGGATCCGTTAACGGTTACCA",
    "GGATCCTTACGAAGCTTGGAACCT",
    "CCTTAAGGCATTACGTTGCAAGGT",
    "AGGTCCTTAACGGATCCTTGCAAT",
    "TTGGAACCTAGGTTCAAGGCATTC",
    "CATTGGAACCTTAGGCAATTCCGA",
    "GCAATTCCGATTAGGAACCTTGGA",
]

# Relative abundances: a realistic skewed library, plus one dropout.
WEIGHTS = [30, 22, 16, 12, 9, 6, 5, 0]

BASES = "ACGT"


def random_dna(rng: random.Random, length: int) -> str:
    return "".join(rng.choice(BASES) for _ in range(length))


def add_errors(rng: random.Random, sequence: str, rate: float) -> str:
    """Apply substitutions, insertions and deletions at roughly ``rate``.

    Indels are given the larger share because they dominate nanopore error.
    """
    out = []
    for base in sequence:
        roll = rng.random()
        if roll < rate * 0.4:                       # substitution
            out.append(rng.choice([b for b in BASES if b != base]))
        elif roll < rate * 0.7:                     # deletion
            continue
        elif roll < rate:                           # insertion
            out.append(base)
            out.append(rng.choice(BASES))
        else:
            out.append(base)
    return "".join(out)


def reverse_complement(seq: str) -> str:
    return seq.translate(str.maketrans("ACGT", "TGCA"))[::-1]


def quality_string(rng: random.Random, length: int, centre: int) -> str:
    """Phred+33 string with per-base scores scattered around ``centre``."""
    scores = [
        max(2, min(50, int(rng.gauss(centre, 4.0))))
        for _ in range(length)
    ]
    return "".join(chr(s + 33) for s in scores)


def build_sample(
    rng: random.Random,
    n_library_reads: int,
    weights: list[int],
    *,
    n_contaminant: int = 0,
    n_lowq: int = 0,
    n_concatemer: int = 0,
    n_no_r: int = 0,
    n_dimer: int = 0,
    n_readthrough: int = 0,
    error_rate: float = 0.012,
    prefix: str = "read",
) -> list[tuple[str, str, str]]:
    """Build one sample's worth of records as ``(id, sequence, quality)``."""
    records: list[tuple[str, str, str]] = []

    def emit(read_id: str, sequence: str, centre: int) -> None:
        records.append((read_id, sequence, quality_string(rng, len(sequence), centre)))

    def amplicon(insert: str) -> str:
        return (
            random_dna(rng, rng.randint(15, 40))
            + FORWARD_FLANK + insert + REVERSE_FLANK
            + random_dna(rng, rng.randint(15, 40))
        )

    for i in range(n_library_reads):
        insert = rng.choices(LIBRARY, weights=weights, k=1)[0]
        read = add_errors(rng, amplicon(insert), rate=error_rate)
        if rng.random() < 0.5:                        # half the reads flip strand
            read = reverse_complement(read)
        emit(f"{prefix}_{i:04d}_lib", read, centre=24)

    contaminant = "TTTTTTTTGGGGGGGGCCCCCCCC"
    for i in range(n_contaminant):
        emit(f"{prefix}_{i:04d}_contaminant",
             add_errors(rng, amplicon(contaminant), rate=error_rate), centre=24)

    for i in range(n_lowq):
        emit(f"{prefix}_{i:04d}_lowq",
             add_errors(rng, amplicon(rng.choice(LIBRARY)), rate=0.09), centre=11)

    for i in range(n_concatemer):
        unit = FORWARD_FLANK + LIBRARY[0] + REVERSE_FLANK
        emit(f"{prefix}_{i:04d}_concatemer",
             add_errors(rng, random_dna(rng, 20) + unit + random_dna(rng, 12)
                        + unit + random_dna(rng, 20), rate=error_rate), centre=24)

    for i in range(n_no_r):
        emit(f"{prefix}_{i:04d}_no_R",
             add_errors(rng, random_dna(rng, 25) + FORWARD_FLANK + LIBRARY[1]
                        + random_dna(rng, 30), rate=error_rate), centre=24)

    for i in range(n_dimer):
        emit(f"{prefix}_{i:04d}_dimer",
             add_errors(rng, random_dna(rng, 25) + FORWARD_FLANK + REVERSE_FLANK
                        + random_dna(rng, 25), rate=error_rate), centre=24)

    for i in range(n_readthrough):
        emit(f"{prefix}_{i:04d}_readthrough",
             add_errors(rng, amplicon(random_dna(rng, 900)), rate=error_rate), centre=24)

    rng.shuffle(records)
    return records


def as_fastq(records: list[tuple[str, str, str]]) -> str:
    return "".join(f"@{rid}\n{seq}\n+\n{qual}\n" for rid, seq, qual in records)


def gzip_deterministically(data: bytes) -> bytes:
    """Gzip ``data`` without stamping the current time into the header.

    gzip records an mtime by default, so re-running this script would produce
    different bytes for identical content -- which would break the promise
    that the checked-in example files regenerate byte-for-byte.  (GzipFile is
    used rather than gzip.compress(mtime=...) purely for clarity about what is
    being suppressed.)
    """
    buffer = io.BytesIO()
    with gzip.GzipFile(fileobj=buffer, mode="wb", mtime=0) as handle:
        handle.write(data)
    return buffer.getvalue()


def main() -> None:
    rng = random.Random(SEED)

    # --- the single-sample quickstart dataset -------------------------------
    records = build_sample(
        rng, n_library_reads=450, weights=WEIGHTS,
        n_contaminant=25, n_lowq=40, n_concatemer=8,
        n_no_r=6, n_dimer=5, n_readthrough=4,
    )
    fastq = HERE / "example_reads.fastq"
    fastq.write_text(as_fastq(records))
    (HERE / "example_library.txt").write_text(
        "# intended library: 8 designed 24 bp barcodes\n"
        + "\n".join(LIBRARY) + "\n"
    )
    print(f"wrote {len(records)} reads to {fastq}")

    # --- a four-barcode batch, each sample behaving differently -------------
    # barcode01  a clean, deep sample
    # barcode02  shallower, otherwise normal
    # barcode03  heavily contaminated
    # barcode04  two library members dropped out
    batch_specs = [
        ("barcode01", dict(n_library_reads=300, weights=WEIGHTS, n_contaminant=8,
                           n_lowq=20, n_concatemer=5, n_no_r=4, n_dimer=3)),
        ("barcode02", dict(n_library_reads=120, weights=WEIGHTS, n_contaminant=4,
                           n_lowq=10, n_concatemer=2, n_dimer=2)),
        ("barcode03", dict(n_library_reads=150, weights=WEIGHTS, n_contaminant=90,
                           n_lowq=15, n_readthrough=4)),
        ("barcode04", dict(n_library_reads=220,
                           weights=[30, 22, 0, 12, 9, 0, 5, 0],
                           n_contaminant=5, n_lowq=12, n_no_r=3)),
    ]
    batch_dir = HERE / "batch"
    batch_dir.mkdir(exist_ok=True)
    written = []
    for name, spec in batch_specs:
        sample_records = build_sample(rng, prefix=name, **spec)
        path = batch_dir / f"{name}.fastq.gz"
        path.write_bytes(gzip_deterministically(as_fastq(sample_records).encode("utf-8")))
        written.append(path)
        print(f"wrote {len(sample_records):>4} reads to {path}")

    archive = HERE / "example_batch.zip"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in written:
            # A fixed member timestamp, for the same reason as the gzip mtime.
            info = zipfile.ZipInfo(
                f"plasmidsaurus_order/{path.name}", date_time=(1980, 1, 1, 0, 0, 0)
            )
            info.compress_type = zipfile.ZIP_DEFLATED
            zf.writestr(info, path.read_bytes())
    print(f"wrote {archive}")

    print(f"F = {FORWARD_FLANK}")
    print(f"R = {REVERSE_FLANK}")


if __name__ == "__main__":
    main()
