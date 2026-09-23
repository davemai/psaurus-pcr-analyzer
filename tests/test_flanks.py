"""Step 2: fuzzy flank matching, strand handling and every edge case."""

from __future__ import annotations

import pytest
from conftest import (
    F,
    LIB_A,
    LIB_B,
    PAD_LEFT,
    PAD_RIGHT,
    R,
    amplicon,
    mutate,
)

from psaurus_pcr.config import AnalysisParams
from psaurus_pcr.flanks import (
    STATUS_EMPTY_INSERT,
    STATUS_EXTRACTED,
    STATUS_NO_FLANKS,
    STATUS_NO_FORWARD,
    STATUS_NO_REVERSE,
    STATUS_TOO_LONG,
    STATUS_WRONG_ORDER,
    extract_insert,
    find_flank,
    orient_reverse_flank,
)
from psaurus_pcr.sequtils import reverse_complement

BIG = 10_000  # effectively "no max insert length" for these unit tests


def run(sequence, params=None, max_insert=BIG, reverse=R, read_id="read"):
    return extract_insert(
        read_id, sequence, F, reverse, params or AnalysisParams(), max_insert
    )


# --- the happy path ---------------------------------------------------------


def test_clean_exact_match_is_extracted():
    result = run(amplicon(LIB_A))
    assert result.status == STATUS_EXTRACTED
    assert result.insert == LIB_A
    assert result.orientation == "forward"
    assert result.forward_edit_distance == 0
    assert result.reverse_edit_distance == 0
    assert result.ambiguous is False
    assert result.counted is True


def test_insert_is_strictly_between_the_flanks():
    """No part of F or R may leak into the extracted insert."""
    result = run(amplicon(LIB_A))
    assert not result.insert.startswith(F[-3:])
    assert not result.insert.endswith(R[:3])
    assert F not in result.insert and R not in result.insert


# --- fuzzy tolerance --------------------------------------------------------


def test_match_requiring_fuzzy_tolerance():
    """Two substitutions in each flank: within the default 12% budget for 20 bp."""
    noisy = PAD_LEFT + mutate(F, [3, 11]) + LIB_A + mutate(R, [5, 14]) + PAD_RIGHT
    result = run(noisy)
    assert result.status == STATUS_EXTRACTED
    assert result.insert == LIB_A
    assert result.forward_edit_distance == 2
    assert result.reverse_edit_distance == 2


def test_indel_in_flank_is_tolerated():
    """Nanopore's characteristic error is an indel, not a substitution."""
    deleted = F[:8] + F[9:]  # 19 bp: one base deleted from the middle of F
    result = run(PAD_LEFT + deleted + LIB_A + R + PAD_RIGHT)
    assert result.status == STATUS_EXTRACTED
    assert result.insert == LIB_A
    assert result.forward_edit_distance == 1


def test_exact_matching_rejects_a_read_fuzzy_matching_accepts():
    noisy = PAD_LEFT + mutate(F, [3, 11]) + LIB_A + R + PAD_RIGHT
    strict = AnalysisParams(forward_max_edits=0, reverse_max_edits=0)
    assert run(noisy, strict).status == STATUS_NO_FORWARD
    assert run(noisy).status == STATUS_EXTRACTED


def test_too_many_errors_for_the_budget_is_not_a_match():
    very_noisy = PAD_LEFT + mutate(F, [1, 4, 7, 10, 13, 16]) + LIB_A + R + PAD_RIGHT
    assert run(very_noisy).status == STATUS_NO_FORWARD


# --- strandedness -----------------------------------------------------------


def test_reverse_complement_read_is_found_and_reported_in_f_to_r_orientation():
    """~Half of nanopore reads are the reverse complement; both must collapse together."""
    forward_read = amplicon(LIB_A)
    reverse_read = reverse_complement(forward_read)

    forward_result = run(forward_read)
    reverse_result = run(reverse_read)

    assert reverse_result.status == STATUS_EXTRACTED
    assert reverse_result.orientation == "reverse"
    # The key property: both orientations yield the *same* insert string, so
    # exact-match collapsing in step 3 is meaningful.
    assert reverse_result.insert == forward_result.insert == LIB_A


def test_r_convention_revcomp_handles_a_wet_lab_reverse_primer():
    r_as_primer = reverse_complement(R)
    read = amplicon(LIB_A)
    # Given literally, a reverse primer is not on the F strand and finds nothing.
    assert run(read, reverse=r_as_primer).status == STATUS_NO_REVERSE
    # Reverse-complemented, it is exactly the site we expect downstream of F.
    searched = orient_reverse_flank(r_as_primer, "revcomp")
    assert run(read, reverse=searched).status == STATUS_EXTRACTED


# --- missing / mis-ordered flanks ------------------------------------------


def test_missing_reverse_flank():
    result = run(PAD_LEFT + F + LIB_A + PAD_RIGHT)
    assert result.status == STATUS_NO_REVERSE
    assert result.insert is None
    assert result.counted is False
    assert result.forward_edit_distance == 0


def test_missing_forward_flank():
    result = run(PAD_LEFT + LIB_A + R + PAD_RIGHT)
    assert result.status == STATUS_NO_FORWARD
    assert result.reverse_edit_distance == 0


def test_no_flanks_at_all():
    assert run(PAD_LEFT + LIB_A + PAD_RIGHT).status == STATUS_NO_FLANKS


def test_flanks_in_the_wrong_order():
    """R...F instead of F...R -- an inverted or chimeric product."""
    result = run(PAD_LEFT + R + LIB_A + F + PAD_RIGHT)
    assert result.status == STATUS_WRONG_ORDER
    assert result.insert is None


def test_overlapping_flanks_are_not_an_extraction():
    """When the two flank hits overlap they cannot delimit an insert.

    Uses a deliberately badly-designed primer pair whose sites share 11 bases,
    so both flanks match the read but R starts before F ends.
    """
    bad_f = "AAACCCGGGTTTACGTACGT"
    bad_r = "TTTACGTACGTGGCATTAAC"   # shares F's last 11 bases
    read = PAD_LEFT + "AAACCCGGG" + "TTTACGTACGT" + "GGCATTAAC" + PAD_RIGHT

    result = extract_insert("read", read, bad_f, bad_r, AnalysisParams(), BIG)
    assert result.status == STATUS_WRONG_ORDER
    assert result.forward_edit_distance == 0
    assert result.reverse_edit_distance == 0
    assert result.insert is None
    assert result.counted is False


# --- insert length edge cases ----------------------------------------------


def test_empty_insert_is_flagged_not_counted():
    result = run(PAD_LEFT + F + R + PAD_RIGHT)
    assert result.status == STATUS_EMPTY_INSERT
    assert result.insert == ""
    assert result.insert_length == 0
    assert result.counted is False


def test_absurdly_long_insert_is_flagged_not_counted():
    huge = "ACGT" * 300  # 1200 bp
    result = run(amplicon(huge), max_insert=100)
    assert result.status == STATUS_TOO_LONG
    assert result.insert_length == 1200
    assert result.counted is False
    # The insert is retained on the record so the user can inspect it.
    assert result.insert == huge


def test_min_insert_length_flags_short_products():
    result = run(amplicon("ACG"), AnalysisParams(min_insert_length=10))
    assert result.status == "insert_too_short"
    assert result.counted is False


# --- ambiguity --------------------------------------------------------------


def test_concatemer_gives_multiple_flank_candidates_and_is_flagged():
    """Two tandem copies of the amplicon: a classic read-through product."""
    unit = F + LIB_A + R
    read = PAD_LEFT + unit + "GGATCCTT" + unit + PAD_RIGHT
    result = run(read)

    assert result.status == STATUS_EXTRACTED
    assert result.ambiguous is True
    assert result.forward_ambiguous and result.reverse_ambiguous
    # Best-scoring (and here left-most) hits are used, so the first unit wins.
    assert result.insert == LIB_A
    assert result.counted is True, "ambiguous reads are counted by default"


def test_ambiguous_reads_can_be_excluded_on_request():
    unit = F + LIB_A + R
    read = PAD_LEFT + unit + "GGATCCTT" + unit + PAD_RIGHT
    result = run(read, AnalysisParams(exclude_ambiguous=True))
    assert result.status == "ambiguous_flank_excluded"
    assert result.counted is False


def test_unambiguous_read_is_not_flagged():
    assert run(amplicon(LIB_B)).ambiguous is False


def test_find_flank_reports_the_best_hit_and_its_runner_up():
    near_miss = mutate(F, [2, 6, 9])          # 3 edits away
    target = PAD_LEFT + F + LIB_A + near_miss + PAD_RIGHT
    hit = find_flank(F, target, max_edits=4, ambiguity_margin=1)
    assert hit is not None
    assert hit.edit_distance == 0
    assert target[hit.start : hit.end + 1] == F
    # The rival is 3 edits away, outside the 1-edit margin, so it is neither
    # reported nor treated as a competing candidate.
    assert hit.runner_up_distance is None
    assert hit.ambiguous is False, "3 edits away is not 'near-equally good'"

    # Widen the margin and the same rival becomes a genuine competing candidate.
    wide = find_flank(F, target, max_edits=4, ambiguity_margin=3)
    assert wide.runner_up_distance == 3
    assert wide.ambiguous is True


def test_find_flank_returns_none_when_out_of_budget():
    assert find_flank(F, PAD_LEFT + LIB_A + PAD_RIGHT, max_edits=2) is None


def test_orient_reverse_flank_requires_a_resolved_convention():
    assert orient_reverse_flank(R, "literal") == R
    assert orient_reverse_flank(R, "revcomp") == reverse_complement(R)
    with pytest.raises(ValueError):
        orient_reverse_flank(R, "auto")
