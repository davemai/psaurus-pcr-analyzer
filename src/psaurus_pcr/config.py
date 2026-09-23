"""Configuration objects shared by the core logic, the CLI and the web app.

Everything tunable lives in :class:`AnalysisParams` so that a run is fully
described by (fastq, F, R, library, params).  The CLI builds one of these from
argparse; Streamlit builds one from sidebar widgets; a notebook can build one
by hand.  No analysis function ever reads ``sys.argv``.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Optional

# ---------------------------------------------------------------------------
# Defaults, with the biological reasoning spelled out.
# ---------------------------------------------------------------------------

#: Default mean-read-quality cutoff.
#:
#: Q20 == 1 expected error per 100 bases (99% accuracy).  Historically, ONT
#: pipelines used Q7-Q10 because 1D/R9.4.1 chemistry simply could not do better;
#: much legacy documentation still quotes Q9.  Modern R10.4.1 flowcells basecalled
#: with the "sup" (super-accurate) model -- which is what Plasmidsaurus runs --
#: routinely produce modal read qualities of Q20+, so Q20 is a reasonable
#: "keep the good reads" threshold rather than an aggressive one.  Lower it
#: (e.g. Q12-Q15) if you are losing too many reads on an older or rescued run.
DEFAULT_MIN_MEAN_QUALITY: float = 20.0

#: Default allowed edit distance for a flank, as a fraction of the flank length.
#:
#: Even after Q20 filtering, a read averaging 1% error will carry ~0.2 errors in
#: a 20 bp flank, and errors are not uniformly distributed (homopolymers and
#: methylated motifs are worse).  Requiring an exact flank match therefore
#: throws away a large and *biased* fraction of real reads.  12% of a 20 bp
#: flank == 2 edits, of a 25 bp flank == 3 edits.
#:
#: Tradeoff: too loose and a short flank will match random sequence by chance
#: (a 20 bp query at 5 edits has a non-trivial chance of hitting anywhere in a
#: multi-kb read), inflating counts with garbage inserts; too tight and you drop
#: real reads in proportion to how error-prone the run was.  Tune empirically:
#: sweep this parameter and watch the "extracted" rate plateau.
DEFAULT_FLANK_EDIT_FRACTION: float = 0.12

#: Fallback maximum insert length when no library file is supplied.
DEFAULT_MAX_INSERT_LENGTH: int = 5000

#: Multiplier applied to the longest library sequence to derive the max insert
#: length when a library *is* supplied.  3x leaves room for a genuine
#: read-through / partial concatemer to be flagged rather than counted, while
#: not clipping honest length variation in the library.
DEFAULT_LIBRARY_LENGTH_MULTIPLIER: float = 3.0

#: An alternative flank hit is called "near-equally good" if its edit distance
#: is within this many edits of the best hit.  Such reads are usually
#: concatemers, tandem duplications or read-through products.
DEFAULT_AMBIGUITY_MARGIN: int = 1

#: Number of quality-passing reads used to auto-detect the R flank convention.
DEFAULT_ORIENTATION_PROBE_READS: int = 2000


QUALITY_METRICS = ("mean_phred", "error_prob")
R_CONVENTIONS = ("auto", "literal", "revcomp")
OUTPUT_FORMATS = ("tsv", "csv")


@dataclass
class AnalysisParams:
    """All tunable knobs for one run.

    Attributes
    ----------
    min_mean_quality:
        Reads whose mean quality is strictly below this are discarded.
    quality_metric:
        ``"mean_phred"`` (default) takes the arithmetic mean of the per-base
        Phred scores -- this is what "mean read Q" conventionally means and what
        tools like NanoFilt report.  ``"error_prob"`` instead converts each base
        to an error probability, averages those, and converts back
        (``-10*log10(mean p_err)``); this is how ONT itself computes a read
        qscore and is the more defensible statistic, because Phred is a log
        scale and averaging logs over-weights good bases.  The two differ by a
        few Q units on typical nanopore reads, so **state which you used** when
        you quote a threshold.
    flank_edit_fraction:
        Allowed edits per flank as a fraction of that flank's length.  Ignored
        for a flank whose absolute budget is set explicitly.
    forward_max_edits, reverse_max_edits:
        Absolute edit budgets.  ``None`` means "derive from
        ``flank_edit_fraction``".
    ambiguity_margin:
        A second, non-overlapping hit within this many edits of the best hit
        makes the read "ambiguous" for that flank.
    max_insert_length:
        Inserts longer than this are flagged and excluded from quantification.
        ``None`` means "derive from the library file, else
        :data:`DEFAULT_MAX_INSERT_LENGTH`".
    min_insert_length:
        Inserts shorter than this are flagged and excluded.  The default of 1
        means only truly empty inserts (F immediately abutting R) are dropped.
    exclude_ambiguous:
        If True, reads with an ambiguous flank hit are excluded from the count
        table as well as being reported.  Default False: they are counted using
        the best-scoring hit, but carry an ``ambiguous`` flag and are reported
        separately in QC.
    r_convention:
        How to interpret the R flank string (see :mod:`psaurus_pcr.flanks`).
    fuzzy_library:
        Enable fuzzy library matching (off by default: library members are
        *defined* sequences, so an inexact hit is a finding, not a match).
    library_edit_fraction, library_max_edits:
        Tolerance for fuzzy library matching, fraction-of-length or absolute.
    top_n_plot:
        How many sequences to show in the abundance bar chart.
    """

    # --- Step 1: quality ---------------------------------------------------
    min_mean_quality: float = DEFAULT_MIN_MEAN_QUALITY
    quality_metric: str = "mean_phred"

    # --- Step 2: flank matching -------------------------------------------
    flank_edit_fraction: float = DEFAULT_FLANK_EDIT_FRACTION
    forward_max_edits: Optional[int] = None
    reverse_max_edits: Optional[int] = None
    ambiguity_margin: int = DEFAULT_AMBIGUITY_MARGIN
    r_convention: str = "auto"
    orientation_probe_reads: int = DEFAULT_ORIENTATION_PROBE_READS

    # --- Step 2b: insert sanity checks ------------------------------------
    max_insert_length: Optional[int] = None
    min_insert_length: int = 1
    library_length_multiplier: float = DEFAULT_LIBRARY_LENGTH_MULTIPLIER
    exclude_ambiguous: bool = False

    # --- Step 4: library comparison ---------------------------------------
    fuzzy_library: bool = False
    library_edit_fraction: float = 0.05
    library_max_edits: Optional[int] = None

    # --- reporting ---------------------------------------------------------
    top_n_plot: int = 20

    def __post_init__(self) -> None:
        if self.quality_metric not in QUALITY_METRICS:
            raise ValueError(
                f"quality_metric must be one of {QUALITY_METRICS}, got {self.quality_metric!r}"
            )
        if self.r_convention not in R_CONVENTIONS:
            raise ValueError(
                f"r_convention must be one of {R_CONVENTIONS}, got {self.r_convention!r}"
            )
        if not 0.0 <= self.flank_edit_fraction < 1.0:
            raise ValueError("flank_edit_fraction must be in [0, 1)")
        if self.ambiguity_margin < 0:
            raise ValueError("ambiguity_margin must be >= 0")
        if self.min_insert_length < 0:
            raise ValueError("min_insert_length must be >= 0")
        if self.max_insert_length is not None and self.max_insert_length < 0:
            raise ValueError("max_insert_length must be >= 0")

    # -- derived values ------------------------------------------------------

    def edits_for_flank(self, flank: str, which: str) -> int:
        """Return the edit budget for ``flank``.

        ``which`` is ``"forward"`` or ``"reverse"``.  An explicit absolute
        budget wins; otherwise the budget is ``round(fraction * len(flank))``,
        floored at 0.  We use ``floor`` semantics via ``int()`` on the product
        so that a 10 bp flank at 12% gets 1 edit, not 0 -- see below.
        """
        explicit = self.forward_max_edits if which == "forward" else self.reverse_max_edits
        if explicit is not None:
            return max(0, int(explicit))
        # round() gives 20bp -> 2, 25bp -> 3, 30bp -> 4 at the 12% default,
        # which matches the "10-15% of flank length" rule of thumb.
        return max(0, int(round(self.flank_edit_fraction * len(flank))))

    def library_edits_for(self, sequence: str) -> int:
        """Edit budget for fuzzy-matching one library sequence."""
        if self.library_max_edits is not None:
            return max(0, int(self.library_max_edits))
        return max(1, int(math.ceil(self.library_edit_fraction * len(sequence))))

    def to_dict(self) -> dict:
        """JSON-serialisable view, for the run summary."""
        return asdict(self)
