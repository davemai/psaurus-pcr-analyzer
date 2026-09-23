"""Step 3 -- collapse extracted inserts into unique sequences and count them.

Collapsing is by **exact string identity** of the extracted insert.  That is
the right default for a defined-library PCR experiment: any residual basecall
error shows up as its own low-count "unique" sequence, which is informative
(it tells you the error rate) rather than being silently merged into a
neighbour.  It also means the singleton fraction is a useful diagnostic -- a
high singleton rate means either a noisy run or a genuinely diverse library.

Because step 2 always returns the insert in the F->R orientation, reads from
both strands collapse onto the same key.
"""

from __future__ import annotations

from collections import Counter
from typing import Iterable

import pandas as pd

from psaurus_pcr.flanks import ExtractionResult
from psaurus_pcr.sequtils import summarise_numeric, summarise_weighted

UNIQUE_COLUMNS = ["sequence", "count", "percent_of_extracted", "length"]


def count_inserts(extractions: Iterable[ExtractionResult]) -> Counter:
    """Counter over the inserts of reads that passed every step-2 check."""
    counter: Counter = Counter()
    for result in extractions:
        if result.counted and result.insert is not None:
            counter[result.insert] += 1
    return counter


def counts_to_dataframe(counter: Counter) -> pd.DataFrame:
    """Sorted table of unique sequences.

    Sorted by descending count, then ascending sequence so that the output is
    deterministic when counts tie (important for diffing runs and for tests).
    """
    total = sum(counter.values())
    rows = [
        {
            "sequence": seq,
            "count": count,
            "percent_of_extracted": (100.0 * count / total) if total else 0.0,
            "length": len(seq),
        }
        for seq, count in counter.items()
    ]
    frame = pd.DataFrame(rows, columns=UNIQUE_COLUMNS)
    if frame.empty:
        return frame
    frame = frame.sort_values(
        ["count", "sequence"], ascending=[False, True], kind="mergesort"
    ).reset_index(drop=True)
    frame["percent_of_extracted"] = frame["percent_of_extracted"].round(6)
    return frame


def quantify(extractions: Iterable[ExtractionResult]) -> pd.DataFrame:
    """Convenience: :func:`count_inserts` followed by :func:`counts_to_dataframe`."""
    return counts_to_dataframe(count_inserts(extractions))


def quantification_summary(frame: pd.DataFrame) -> dict:
    """Summary statistics for the unique-sequence table (JSON-serialisable)."""
    if frame.empty:
        return {
            "total_extracted_reads": 0,
            "unique_sequences": 0,
            "singletons": 0,
            "percent_singletons": 0.0,
            "top_sequence_count": 0,
            "percent_reads_in_top_sequence": 0.0,
            # Both keys are emitted even when empty, so every sample in a batch
            # contributes the same set of metrics to the combined QC table.
            "insert_length": summarise_numeric([]),
            "insert_length_unique_sequences": summarise_numeric([]),
        }
    counts = frame["count"]
    total = int(counts.sum())
    singletons = int((counts == 1).sum())
    # Length distribution weighted by read support, i.e. the length
    # distribution of extracted *reads*, not of distinct sequences.  Computed
    # from (length, count) pairs rather than an expanded per-read list, which
    # would be O(reads) in memory on a deep run.
    weighted_lengths = zip(frame["length"], counts)
    return {
        "total_extracted_reads": total,
        "unique_sequences": int(len(frame)),
        "singletons": singletons,
        "percent_singletons": round(100.0 * singletons / len(frame), 4),
        "top_sequence_count": int(counts.iloc[0]),
        "percent_reads_in_top_sequence": round(100.0 * int(counts.iloc[0]) / total, 4),
        "insert_length": summarise_weighted(weighted_lengths),
        "insert_length_unique_sequences": summarise_numeric(
            [int(x) for x in frame["length"]]
        ),
    }
