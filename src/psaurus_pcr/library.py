"""Step 4 -- compare the observed unique sequences against an intended library.

Matching policy
---------------
**Exact match is the default.**  A library is a set of *designed* sequences, so
"close but not identical" is a result (synthesis error, basecall error,
recombination), not a match.  Collapsing near-misses into their intended
neighbour by default would hide exactly the thing you usually want to see.

Before comparing, both sides are normalised: surrounding and internal
whitespace removed, upper-cased.  That is robustness to *formatting*, not to
sequence -- a normalised comparison is still an exact sequence comparison.

**Fuzzy matching is opt-in** (``--fuzzy-library`` / the sidebar toggle).  It
uses global (Needleman-Wunsch) edit distance via :mod:`edlib`, with a budget of
``library_edit_fraction`` of the library sequence's length (default 5%, at
least 1 edit) or an explicit ``library_max_edits``.  Each observed sequence is
assigned to its single best library member within budget; ties go to the
earliest library entry.  Exact matches always take precedence over fuzzy ones.

Note that no reverse-complement matching is attempted: step 2 already
orientates every insert F->R, so a library member that only matches in reverse
complement indicates the library file is written on the opposite strand, which
is worth surfacing rather than papering over.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import edlib
import pandas as pd

from psaurus_pcr.config import AnalysisParams
from psaurus_pcr.sequtils import VALID_DNA, normalise_sequence

LIBRARY_COLUMNS = [
    "library_id",
    "library_sequence",
    "library_length",
    "detected",
    "count",
    "percent_of_extracted",
    "match_type",
    "n_observed_variants",
    "best_observed_sequence",
    "best_edit_distance",
]


@dataclass
class LibraryComparison:
    """Everything step 4 produces."""

    library_table: pd.DataFrame
    annotated_unique: pd.DataFrame
    summary: dict


def prepare_library(
    sequences: Sequence[str],
) -> Tuple[List[str], List[str], int, List[str]]:
    """Normalise, validate, drop blanks and de-duplicate library entries.

    Returns ``(ids, sequences, n_duplicates_removed, rejected_entries)``.  IDs
    are positional (``lib_0001`` ...) because a bare one-sequence-per-line file
    carries no names; the numbering follows the original file order.

    Entries that are not DNA are **rejected rather than kept**.  A column
    pasted out of a spreadsheet often carries its header ("sequence",
    "barcode_id"), and a comment or header line survives when a caller passes
    a Python list instead of a file.  Keeping such an entry would add a library
    member that can never be detected, quietly deflating "% of library
    detected" -- so they are dropped and reported instead.
    """
    ids: List[str] = []
    cleaned: List[str] = []
    seen: Dict[str, str] = {}
    duplicates = 0
    rejected: List[str] = []
    for index, raw in enumerate(sequences, start=1):
        text = str(raw).strip()
        if not text or text.startswith("#") or text.startswith(">"):
            continue
        seq = normalise_sequence(text)
        if not seq:
            continue
        if set(seq) - VALID_DNA:
            rejected.append(text[:60])
            continue
        if seq in seen:
            duplicates += 1
            continue
        lib_id = f"lib_{index:04d}"
        seen[seq] = lib_id
        ids.append(lib_id)
        cleaned.append(seq)
    return ids, cleaned, duplicates, rejected


def _best_fuzzy_match(
    observed: str,
    library_sequences: Sequence[str],
    params: AnalysisParams,
) -> Optional[Tuple[int, int]]:
    """Return ``(library_index, edit_distance)`` for the best in-budget match."""
    best_index: Optional[int] = None
    best_distance: Optional[int] = None
    observed_length = len(observed)
    for index, candidate in enumerate(library_sequences):
        budget = params.library_edits_for(candidate)
        if best_distance is not None:
            budget = min(budget, best_distance - 1)
            if budget < 0:
                continue
        # Global edit distance is at least the length difference, so this
        # rejects most of the library without calling the aligner at all --
        # the difference between seconds and minutes on a big design.
        if abs(observed_length - len(candidate)) > budget:
            continue
        # Global alignment: the whole observed insert must correspond to the
        # whole library member, so a truncated product is not a "match".
        result = edlib.align(observed, candidate, mode="NW", task="distance", k=budget)
        distance = int(result["editDistance"])
        if distance < 0:
            continue
        if best_distance is None or distance < best_distance:
            best_index, best_distance = index, distance
    if best_index is None or best_distance is None:
        return None
    return best_index, best_distance


def compare_to_library(
    unique_frame: pd.DataFrame,
    library_sequences: Sequence[str],
    params: AnalysisParams | None = None,
) -> LibraryComparison:
    """Annotate observed sequences with library membership and summarise coverage.

    ``unique_frame`` is the table from :func:`psaurus_pcr.quantify.quantify`.
    Nothing is written to disk; both returned tables are plain DataFrames.
    """
    params = params or AnalysisParams()
    lib_ids, lib_seqs, duplicates_removed, rejected = prepare_library(library_sequences)
    index_by_sequence = {seq: i for i, seq in enumerate(lib_seqs)}

    annotated = unique_frame.copy()
    total_reads = int(annotated["count"].sum()) if not annotated.empty else 0

    match_types: List[str] = []
    match_ids: List[Optional[str]] = []
    match_seqs: List[Optional[str]] = []
    match_dists: List[Optional[int]] = []

    # Per-library-member accumulators.
    per_library_reads = [0] * len(lib_seqs)
    per_library_variants = [0] * len(lib_seqs)
    per_library_best: List[Optional[Tuple[int, str, int]]] = [None] * len(lib_seqs)
    per_library_type: List[Optional[str]] = [None] * len(lib_seqs)
    per_library_best_rank: List[Optional[Tuple[int, int]]] = [None] * len(lib_seqs)

    observed_sequences = list(annotated["sequence"]) if not annotated.empty else []
    observed_counts = list(annotated["count"]) if not annotated.empty else []

    for observed, count in zip(observed_sequences, observed_counts):
        count = int(count)
        index = index_by_sequence.get(observed)
        distance: Optional[int] = None
        kind = "none"
        if index is not None:
            kind, distance = "exact", 0
        elif params.fuzzy_library and lib_seqs:
            hit = _best_fuzzy_match(observed, lib_seqs, params)
            if hit is not None:
                index, distance = hit
                kind = "fuzzy"

        if index is None:
            match_types.append("none")
            match_ids.append(None)
            match_seqs.append(None)
            match_dists.append(None)
            continue

        match_types.append(kind)
        match_ids.append(lib_ids[index])
        match_seqs.append(lib_seqs[index])
        match_dists.append(distance)

        per_library_reads[index] += count
        per_library_variants[index] += 1
        # "Best observed" = the most abundant supporting variant; exact beats fuzzy.
        rank = (0 if kind == "exact" else 1, -count)
        if per_library_best_rank[index] is None or rank < per_library_best_rank[index]:
            per_library_best_rank[index] = rank
            per_library_best[index] = (count, observed, distance or 0)
            per_library_type[index] = kind

    if not annotated.empty:
        annotated["library_match_type"] = match_types
        annotated["library_id"] = match_ids
        annotated["library_sequence"] = match_seqs
        # Nullable integer: "no match" is missing, not zero edits.
        annotated["library_edit_distance"] = pd.array(match_dists, dtype="Int64")
        annotated["in_library"] = [m != "none" for m in match_types]
    else:
        for column in [
            "library_match_type",
            "library_id",
            "library_sequence",
            "library_edit_distance",
            "in_library",
        ]:
            annotated[column] = pd.Series(dtype="object")

    rows = []
    for i, (lib_id, seq) in enumerate(zip(lib_ids, lib_seqs)):
        reads = per_library_reads[i]
        best = per_library_best[i]
        rows.append(
            {
                "library_id": lib_id,
                "library_sequence": seq,
                "library_length": len(seq),
                "detected": reads > 0,
                "count": reads,
                "percent_of_extracted": round(100.0 * reads / total_reads, 6)
                if total_reads
                else 0.0,
                "match_type": per_library_type[i] or "none",
                "n_observed_variants": per_library_variants[i],
                "best_observed_sequence": best[1] if best else None,
                "best_edit_distance": best[2] if best else None,
            }
        )
    library_table = pd.DataFrame(rows, columns=LIBRARY_COLUMNS)
    if not library_table.empty:
        library_table["best_edit_distance"] = library_table["best_edit_distance"].astype(
            "Int64"
        )
        library_table = library_table.sort_values(
            ["count", "library_id"], ascending=[False, True], kind="mergesort"
        ).reset_index(drop=True)

    detected = int(library_table["detected"].sum()) if not library_table.empty else 0
    n_library = len(lib_seqs)
    unexpected_mask = (
        ~annotated["in_library"].astype(bool) if not annotated.empty else None
    )
    n_unexpected_unique = int(unexpected_mask.sum()) if unexpected_mask is not None else 0
    unexpected_reads = (
        int(annotated.loc[unexpected_mask, "count"].sum()) if unexpected_mask is not None else 0
    )
    n_unique = int(len(annotated))

    summary = {
        "library_sequences_provided": len(library_sequences),
        "library_sequences_unique": n_library,
        "library_duplicate_entries_removed": duplicates_removed,
        "library_entries_rejected": len(rejected),
        "library_entries_rejected_examples": rejected[:5],
        "library_sequences_detected": detected,
        "percent_library_detected": round(100.0 * detected / n_library, 4) if n_library else 0.0,
        "library_sequences_not_detected": n_library - detected,
        "unexpected_unique_sequences": n_unexpected_unique,
        "percent_unique_sequences_unexpected": round(100.0 * n_unexpected_unique / n_unique, 4)
        if n_unique
        else 0.0,
        "reads_in_unexpected_sequences": unexpected_reads,
        "percent_reads_unexpected": round(100.0 * unexpected_reads / total_reads, 4)
        if total_reads
        else 0.0,
        "reads_in_library_sequences": total_reads - unexpected_reads,
        "percent_reads_in_library": round(
            100.0 * (total_reads - unexpected_reads) / total_reads, 4
        )
        if total_reads
        else 0.0,
        "matching_mode": "fuzzy" if params.fuzzy_library else "exact",
        "fuzzy_edit_fraction": params.library_edit_fraction if params.fuzzy_library else None,
        "fuzzy_max_edits": params.library_max_edits if params.fuzzy_library else None,
    }

    return LibraryComparison(
        library_table=library_table, annotated_unique=annotated, summary=summary
    )


def longest_library_sequence(library_sequences: Sequence[str]) -> int:
    """Length of the longest *valid* library entry (0 if there are none).

    Uses the same validation as :func:`prepare_library` so that a rejected
    entry cannot inflate the derived maximum insert length.
    """
    _, cleaned, _, _ = prepare_library(library_sequences)
    return max((len(s) for s in cleaned), default=0)
