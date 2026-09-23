"""Step 2 -- fuzzy flank matching and insert extraction.

Design notes (the biology behind the code)
------------------------------------------
**Degenerate primers.**  Flanks may contain IUPAC ambiguity codes (``N``,
``R``, ``Y``, ...); the aligner is given the corresponding equivalences, so an
``N`` matches any base rather than burning an edit.  The same applies to an
``N`` in a read.

**Why fuzzy matching.**  Even Q20 reads carry ~1% residual error, concentrated
in homopolymers and certain sequence contexts.  Demanding an exact match to a
20-25 bp primer would drop a large, *sequence-biased* subset of otherwise
perfect reads.  We therefore use :mod:`edlib`, which computes true Levenshtein
(substitutions **and** indels -- indels being the characteristic nanopore
error) in "HW" / infix mode: the flank is aligned as a substring of the read
with free end gaps.

**Why both strands.**  A double-stranded PCR product enters the pore from
either end, so a run yields both orientations in roughly equal proportion.  We
search the read as given and its reverse complement, and extract from whichever
orientation produced a valid F...R pair.  The extracted insert is therefore
always reported in the F->R orientation, which is what makes exact-match
collapsing in step 3 meaningful.

**Why flag ambiguity instead of collapsing it.**  A second, near-equally-good
hit for the same flank usually means a concatemer, a tandem duplication, or a
read-through product -- real artefacts worth knowing about.  We take the
best-scoring hit (as specified) but record the fact.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import edlib

from psaurus_pcr.config import AnalysisParams
from psaurus_pcr.sequtils import IUPAC_EQUALITIES, reverse_complement

# --- read classification statuses -----------------------------------------
STATUS_EXTRACTED = "extracted"
STATUS_NO_FLANKS = "no_flanks_found"
STATUS_NO_FORWARD = "forward_flank_not_found"
STATUS_NO_REVERSE = "reverse_flank_not_found"
STATUS_WRONG_ORDER = "flanks_wrong_order_or_overlapping"
STATUS_EMPTY_INSERT = "insert_empty"
STATUS_TOO_SHORT = "insert_too_short"
STATUS_TOO_LONG = "insert_too_long"
STATUS_AMBIGUOUS_EXCLUDED = "ambiguous_flank_excluded"

#: Statuses whose inserts are *not* carried into quantification.
NON_COUNTED_STATUSES = (
    STATUS_NO_FLANKS,
    STATUS_NO_FORWARD,
    STATUS_NO_REVERSE,
    STATUS_WRONG_ORDER,
    STATUS_EMPTY_INSERT,
    STATUS_TOO_SHORT,
    STATUS_TOO_LONG,
    STATUS_AMBIGUOUS_EXCLUDED,
)

ALL_STATUSES = (STATUS_EXTRACTED,) + NON_COUNTED_STATUSES

# Ranking used to pick the more informative of the two orientations when
# neither yields a usable insert (lower == more informative).
_OUTCOME_RANK = {
    STATUS_EXTRACTED: 0,
    STATUS_WRONG_ORDER: 1,
    STATUS_NO_REVERSE: 2,
    STATUS_NO_FORWARD: 2,
    STATUS_NO_FLANKS: 3,
}


@dataclass(frozen=True)
class FlankHit:
    """One flank alignment against a read.

    ``start``/``end`` are 0-based, **inclusive** coordinates on the searched
    sequence (edlib's convention), so the flank occupies ``seq[start:end + 1]``.

    ``runner_up_distance`` is the edit distance of the best *other*,
    non-overlapping placement of the same flank -- but only searched for within
    ``ambiguity_margin`` edits of the best hit, since a more distant rival is
    not a competing candidate.  It is ``None`` when no such rival exists.
    """

    start: int
    end: int
    edit_distance: int
    ambiguous: bool = False
    runner_up_distance: Optional[int] = None

    def shifted(self, offset: int) -> "FlankHit":
        """Translate coordinates back onto the parent sequence."""
        return FlankHit(
            start=self.start + offset,
            end=self.end + offset,
            edit_distance=self.edit_distance,
            ambiguous=self.ambiguous,
            runner_up_distance=self.runner_up_distance,
        )


def _merge_locations(locations: Sequence[Tuple[Optional[int], int]]) -> List[Tuple[int, int]]:
    """Collapse edlib's optimal-score locations into disjoint intervals.

    edlib commonly reports several adjacent end positions for what is really
    one alignment (e.g. a trailing gap can be placed in more than one way).
    Those are the same hit, not two candidates, so we merge anything that
    overlaps or abuts before counting candidates.
    """
    cleaned = [(int(s), int(e)) for s, e in locations if s is not None]
    if not cleaned:
        return []
    cleaned.sort()
    merged = [cleaned[0]]
    for start, end in cleaned[1:]:
        last_start, last_end = merged[-1]
        if start <= last_end + 1:
            merged[-1] = (last_start, max(last_end, end))
        else:
            merged.append((start, end))
    return merged


def _best_distance(query: str, target: str, max_edits: int) -> int:
    """Best infix edit distance of ``query`` in ``target``, or -1 if > ``max_edits``."""
    if not target or not query:
        return -1
    result = edlib.align(
        query, target, mode="HW", task="distance", k=max_edits,
        additionalEqualities=IUPAC_EQUALITIES,
    )
    return int(result["editDistance"])


def find_flank(
    query: str,
    target: str,
    max_edits: int,
    ambiguity_margin: int = 1,
) -> Optional[FlankHit]:
    """Locate ``query`` within ``target`` allowing up to ``max_edits`` edits.

    Returns the best-scoring (lowest edit distance) hit, ties broken by
    left-most position, or ``None`` if nothing scores within the budget.

    The hit is marked ``ambiguous`` when a *second, non-overlapping* placement
    of the flank scores within ``ambiguity_margin`` edits of the best one --
    the concatemer / read-through signal described in the module docstring.
    """
    if not query or not target:
        return None

    result = edlib.align(
        query, target, mode="HW", task="locations", k=max_edits,
        additionalEqualities=IUPAC_EQUALITIES,
    )
    best = int(result["editDistance"])
    if best < 0:
        return None

    locations = _merge_locations(result.get("locations") or [])
    if not locations:
        return None
    start, end = locations[0]  # leftmost among equally-scoring placements

    runner_up: Optional[int] = None
    if len(locations) > 1:
        # Two disjoint placements already tie for best -> unambiguously ambiguous.
        runner_up = best
    else:
        # Look for the next-best placement outside the best hit.  Searching the
        # flanking segments (rather than masking) avoids inventing hybrid hits
        # that straddle the region we just consumed.
        probe_budget = best + ambiguity_margin
        candidates = [
            _best_distance(query, target[:start], probe_budget),
            _best_distance(query, target[end + 1:], probe_budget),
        ]
        valid = [d for d in candidates if d >= 0]
        if valid:
            runner_up = min(valid)

    ambiguous = runner_up is not None and runner_up <= best + ambiguity_margin
    return FlankHit(
        start=start,
        end=end,
        edit_distance=best,
        ambiguous=ambiguous,
        runner_up_distance=runner_up,
    )


@dataclass
class PairLocation:
    """Outcome of looking for an ordered F...R pair in one orientation."""

    status: str
    forward: Optional[FlankHit] = None
    reverse: Optional[FlankHit] = None

    @property
    def total_edits(self) -> int:
        total = 0
        if self.forward is not None:
            total += self.forward.edit_distance
        if self.reverse is not None:
            total += self.reverse.edit_distance
        return total

    @property
    def rank(self) -> int:
        return _OUTCOME_RANK.get(self.status, 9)


def locate_flank_pair(
    sequence: str,
    forward_flank: str,
    reverse_flank: str,
    forward_max_edits: int,
    reverse_max_edits: int,
    ambiguity_margin: int = 1,
) -> PairLocation:
    """Find F followed downstream by R in a single orientation of one read.

    ``reverse_flank`` must already be given in the orientation in which it is
    expected to appear on the same strand as ``forward_flank`` -- see
    :func:`orient_reverse_flank`.

    Two anchorings are tried so that a spurious best hit for one flank cannot
    hide a genuine pair:

    1. anchor on the best F in the whole read, then find the best R *after* it;
    2. anchor on the best R in the whole read, then find the best F *before* it.

    Whichever yields the lower combined edit distance wins.
    """
    f_global = find_flank(forward_flank, sequence, forward_max_edits, ambiguity_margin)
    r_global = find_flank(reverse_flank, sequence, reverse_max_edits, ambiguity_margin)

    if f_global is None and r_global is None:
        return PairLocation(STATUS_NO_FLANKS)
    if f_global is None:
        return PairLocation(STATUS_NO_FORWARD, reverse=r_global)
    if r_global is None:
        return PairLocation(STATUS_NO_REVERSE, forward=f_global)

    candidates: List[Tuple[FlankHit, FlankHit]] = []

    # (1) F-anchored: search R strictly downstream of the end of F.
    tail_offset = f_global.end + 1
    r_downstream = find_flank(
        reverse_flank, sequence[tail_offset:], reverse_max_edits, ambiguity_margin
    )
    if r_downstream is not None:
        candidates.append((f_global, r_downstream.shifted(tail_offset)))

    # (2) R-anchored: search F strictly upstream of the start of R.
    f_upstream = find_flank(
        forward_flank, sequence[: r_global.start], forward_max_edits, ambiguity_margin
    )
    if f_upstream is not None:
        candidates.append((f_upstream, r_global))

    if not candidates:
        # Both flanks are present but never in the F...R order (or they overlap):
        # typical of inverted/chimeric products and adapter read-through.
        return PairLocation(STATUS_WRONG_ORDER, forward=f_global, reverse=r_global)

    f_hit, r_hit = min(
        candidates,
        key=lambda pair: (
            pair[0].edit_distance + pair[1].edit_distance,
            pair[1].start - pair[0].end,  # prefer the tighter (first) amplicon
        ),
    )
    # Ambiguity is a property of the read, not of the chosen anchoring, so keep
    # the global assessment if either anchoring saw a rival hit.
    f_hit = FlankHit(
        f_hit.start, f_hit.end, f_hit.edit_distance,
        f_hit.ambiguous or f_global.ambiguous,
        f_hit.runner_up_distance if f_hit.runner_up_distance is not None else f_global.runner_up_distance,
    )
    r_hit = FlankHit(
        r_hit.start, r_hit.end, r_hit.edit_distance,
        r_hit.ambiguous or r_global.ambiguous,
        r_hit.runner_up_distance if r_hit.runner_up_distance is not None else r_global.runner_up_distance,
    )
    return PairLocation(STATUS_EXTRACTED, forward=f_hit, reverse=r_hit)


@dataclass
class ExtractionResult:
    """Per-read outcome of step 2."""

    read_id: str
    status: str
    orientation: Optional[str] = None       # "forward" | "reverse"
    insert: Optional[str] = None
    insert_length: Optional[int] = None
    forward_edit_distance: Optional[int] = None
    reverse_edit_distance: Optional[int] = None
    forward_ambiguous: bool = False
    reverse_ambiguous: bool = False
    read_length: Optional[int] = None
    mean_quality: Optional[float] = None

    @property
    def ambiguous(self) -> bool:
        return self.forward_ambiguous or self.reverse_ambiguous

    @property
    def counted(self) -> bool:
        """Whether this read's insert enters the quantification table."""
        return self.status == STATUS_EXTRACTED

    def to_row(self) -> dict:
        return {
            "read_id": self.read_id,
            "status": self.status,
            "orientation": self.orientation,
            "read_length": self.read_length,
            "mean_quality": None if self.mean_quality is None else round(self.mean_quality, 3),
            "insert_length": self.insert_length,
            "forward_edit_distance": self.forward_edit_distance,
            "reverse_edit_distance": self.reverse_edit_distance,
            "ambiguous_flank": self.ambiguous,
            "counted": self.counted,
            "insert": self.insert,
        }


def orient_reverse_flank(reverse_flank: str, convention: str) -> str:
    """Return R in the orientation in which it should appear downstream of F.

    ``convention``:

    * ``"literal"`` -- R was given as it reads on the same strand as F, i.e. the
      amplicon looks like ``5'-F...R-3'``.  Search R as typed.
    * ``"revcomp"`` -- R was given as a conventional reverse PCR primer, i.e.
      5'->3' on the *bottom* strand.  Its binding site on the F strand is
      ``revcomp(R)``, so that is what we search for.

    ``"auto"`` is resolved upstream (see :func:`psaurus_pcr.pipeline`) by trying
    both on a subsample of reads and keeping whichever explains more of them.
    """
    if convention == "literal":
        return reverse_flank
    if convention == "revcomp":
        return reverse_complement(reverse_flank)
    raise ValueError(
        f"orient_reverse_flank() needs a resolved convention, got {convention!r}"
    )


def extract_insert(
    read_id: str,
    sequence: str,
    forward_flank: str,
    reverse_flank_on_f_strand: str,
    params: AnalysisParams,
    max_insert_length: int,
    read_length: Optional[int] = None,
    mean_quality: Optional[float] = None,
) -> ExtractionResult:
    """Extract the sequence strictly between the end of F and the start of R.

    Both the given orientation and the reverse complement of the read are
    searched; the orientation producing a valid ordered pair wins (lower total
    edit distance breaks a tie).  The returned insert is always in the F->R
    orientation.
    """
    f_edits = params.edits_for_flank(forward_flank, "forward")
    r_edits = params.edits_for_flank(reverse_flank_on_f_strand, "reverse")

    attempts = [
        ("forward", sequence),
        ("reverse", reverse_complement(sequence)),
    ]
    located: List[Tuple[str, str, PairLocation]] = []
    for orientation, seq in attempts:
        located.append(
            (
                orientation,
                seq,
                locate_flank_pair(
                    seq,
                    forward_flank,
                    reverse_flank_on_f_strand,
                    f_edits,
                    r_edits,
                    params.ambiguity_margin,
                ),
            )
        )

    # Prefer a real F...R pair; otherwise report the more informative failure.
    orientation, oriented_seq, pair = min(
        located, key=lambda item: (item[2].rank, item[2].total_edits)
    )

    base = ExtractionResult(
        read_id=read_id,
        status=pair.status,
        orientation=orientation if pair.status == STATUS_EXTRACTED else None,
        forward_edit_distance=pair.forward.edit_distance if pair.forward else None,
        reverse_edit_distance=pair.reverse.edit_distance if pair.reverse else None,
        forward_ambiguous=bool(pair.forward and pair.forward.ambiguous),
        reverse_ambiguous=bool(pair.reverse and pair.reverse.ambiguous),
        read_length=read_length if read_length is not None else len(sequence),
        mean_quality=mean_quality,
    )

    if pair.status != STATUS_EXTRACTED:
        return base

    insert = oriented_seq[pair.forward.end + 1 : pair.reverse.start]
    base.insert = insert
    base.insert_length = len(insert)

    # --- insert sanity checks, each reported separately ---------------------
    if len(insert) == 0:
        base.status = STATUS_EMPTY_INSERT
    elif len(insert) < params.min_insert_length:
        base.status = STATUS_TOO_SHORT
    elif len(insert) > max_insert_length:
        # "Absurdly long": almost always a concatemer or a missed internal R.
        base.status = STATUS_TOO_LONG
    elif params.exclude_ambiguous and base.ambiguous:
        base.status = STATUS_AMBIGUOUS_EXCLUDED

    return base
