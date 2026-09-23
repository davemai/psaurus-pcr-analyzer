"""Small sequence helpers (complementation, normalisation, Phred maths)."""

from __future__ import annotations

import bisect
import math
from typing import Iterable, List, Sequence, Tuple

# Full IUPAC complement table, including lowercase, so that we never silently
# mangle an ambiguity code in a user-supplied primer.  '-' and '.' are passed
# through so gapped input does not blow up.
_COMPLEMENT = str.maketrans(
    "ACGTUNRYSWKMBDHVacgtunryswkmbdhv-.",
    "TGCAANYRSWMKVHDBtgcaanyrswmkvhdb-.",
)

#: Characters we are willing to see in a DNA sequence supplied by the user.
VALID_DNA = set("ACGTUNRYSWKMBDHV")

#: IUPAC ambiguity codes and the unambiguous bases each one stands for.
IUPAC_CODES = {
    "A": "A", "C": "C", "G": "G", "T": "T", "U": "T",
    "R": "AG", "Y": "CT", "S": "GC", "W": "AT", "K": "GT", "M": "AC",
    "B": "CGT", "D": "AGT", "H": "ACT", "V": "ACG", "N": "ACGT",
}


def _build_iupac_equalities():
    """Character pairs edlib should treat as equal, in both directions.

    Without these, a degenerate primer (or an ``N`` emitted by the basecaller)
    scores a mismatch at every ambiguous position, quietly eating the whole
    edit budget.  Supplying them makes ``N`` in a flank match any base, ``R``
    match A or G, and so on -- which is what the person who typed the primer
    meant.
    """
    pairs = set()
    for code, bases in IUPAC_CODES.items():
        for base in bases:
            if code != base:
                pairs.add((code, base))
                pairs.add((base, code))
    # Ambiguity codes that overlap are also compatible (e.g. R=AG vs V=ACG
    # share A and G), which matters when a read carries an ambiguity code too.
    for first, first_bases in IUPAC_CODES.items():
        for second, second_bases in IUPAC_CODES.items():
            if first != second and set(first_bases) & set(second_bases):
                pairs.add((first, second))
    return sorted(pairs)


#: Passed to every flank alignment as edlib's ``additionalEqualities``.
IUPAC_EQUALITIES = _build_iupac_equalities()


def reverse_complement(seq: str) -> str:
    """Return the reverse complement of ``seq`` (IUPAC-aware).

    Why this matters here: a nanopore read is sequenced from whichever end of
    the duplex enters the pore first, so roughly half of the reads from a PCR
    product are the reverse complement of the other half.  Searching only the
    as-given orientation would therefore throw away ~50% of the data.
    """
    return seq.translate(_COMPLEMENT)[::-1]


def normalise_sequence(seq: str) -> str:
    """Uppercase and strip *all* whitespace from a sequence string.

    Used for user-typed flanks and for library entries, where stray spaces,
    tabs, line wrapping or lowercase are formatting noise rather than biology.
    """
    return "".join(seq.split()).upper()


def validate_flank(seq: str, name: str) -> str:
    """Normalise a flank and raise a helpful error if it is not usable."""
    cleaned = normalise_sequence(seq)
    if not cleaned:
        raise ValueError(f"{name} flank is empty.")
    bad = sorted(set(cleaned) - VALID_DNA)
    if bad:
        raise ValueError(
            f"{name} flank contains non-DNA character(s) {''.join(bad)!r}. "
            "Expected IUPAC nucleotide codes only."
        )
    return cleaned


def mean_phred(qualities: Sequence[int]) -> float:
    """Arithmetic mean of per-base Phred scores.

    This is the conventional "mean read Q" and what most filtering tools
    report.  Note it is a mean of logs, so it is optimistic relative to
    :func:`error_probability_quality`.
    """
    if not qualities:
        return 0.0
    return sum(qualities) / len(qualities)


def error_probability_quality(qualities: Sequence[int]) -> float:
    """Read qscore computed from the mean per-base error probability.

    ``-10 * log10(mean(10 ** (-Q/10)))``.  This is how ONT's basecallers define
    a read's qscore, and it is the statistically honest way to summarise Phred
    values: a handful of terrible bases drags it down, as it should, whereas an
    arithmetic mean of Phred scores hides them.
    """
    if not qualities:
        return 0.0
    mean_p = sum(10.0 ** (-q / 10.0) for q in qualities) / len(qualities)
    if mean_p <= 0:
        # Every base was reported as error-free; cap at a sane ceiling.
        return 93.0
    return -10.0 * math.log10(mean_p)


def read_quality(qualities: Sequence[int], metric: str = "mean_phred") -> float:
    """Dispatch to the requested per-read quality statistic."""
    if metric == "mean_phred":
        return mean_phred(qualities)
    if metric == "error_prob":
        return error_probability_quality(qualities)
    raise ValueError(f"Unknown quality metric {metric!r}")


def _quantiles_from_sorted(getter, n: int) -> dict:
    """min/q1/median/q3/max from an indexable, ascending sequence of ``n`` items.

    ``getter(k)`` returns the k-th (0-based) value.  Shared by the plain and
    the weighted summarisers so both produce identical numbers.
    """

    def q(p: float) -> float:
        if n == 1:
            return getter(0)
        idx = p * (n - 1)
        lo = int(math.floor(idx))
        hi = int(math.ceil(idx))
        if lo == hi:
            return getter(lo)
        return getter(lo) + (getter(hi) - getter(lo)) * (idx - lo)

    return {
        "n": n,
        "min": getter(0),
        "q1": q(0.25),
        "median": q(0.5),
        "q3": q(0.75),
        "max": getter(n - 1),
    }


def _empty_summary() -> dict:
    return {"n": 0, "min": None, "q1": None, "median": None, "q3": None,
            "max": None, "mean": None}


def summarise_weighted(pairs: Iterable[Tuple[float, int]]) -> dict:
    """Summary statistics for ``(value, count)`` pairs, without expanding them.

    Equivalent to ``summarise_numeric`` over the expanded list, but O(distinct
    values) in memory rather than O(reads).  That matters here: the insert
    lengths of a deep run are described by a handful of distinct lengths with
    large counts, and materialising one entry per read would cost hundreds of
    megabytes on a multi-million-read flowcell.
    """
    items = sorted((float(value), int(count)) for value, count in pairs if int(count) > 0)
    total = sum(count for _, count in items)
    if total == 0:
        return _empty_summary()

    # Cumulative counts let us address the k-th element of the notional
    # expanded array with a binary search instead of building it.
    cumulative: List[int] = []
    running = 0
    for _, count in items:
        running += count
        cumulative.append(running)

    def getter(k: int) -> float:
        return items[bisect.bisect_right(cumulative, k)][0]

    summary = _quantiles_from_sorted(getter, total)
    summary["mean"] = sum(value * count for value, count in items) / total
    return summary


def summarise_numeric(values: Iterable[float]) -> dict:
    """min/max/mean/median/quartiles for a list of numbers (no numpy needed)."""
    vals = sorted(float(v) for v in values)
    n = len(vals)
    if n == 0:
        return _empty_summary()
    summary = _quantiles_from_sorted(vals.__getitem__, n)
    summary["mean"] = sum(vals) / n
    return summary
