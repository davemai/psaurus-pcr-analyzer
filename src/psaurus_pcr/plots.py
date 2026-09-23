"""Matplotlib QC figures.

Every function returns a :class:`matplotlib.figure.Figure` so that Streamlit can
render it with ``st.pyplot`` and the CLI can save it -- neither front end needs
its own plotting code, and nothing here writes a file unless
:func:`save_all_plots` is called.

Chart conventions used throughout (kept deliberately plain and consistent):

* one measure per axes, never a second y-scale;
* the quality-filter comparison is drawn as a *stacked* kept/removed
  histogram rather than two overlaid curves: the passing reads are a strict
  subset of all reads, so stacking shows the partition exactly, with no
  translucent overlap and no series hidden behind another;
* a single-series chart carries no legend (the title names it); a two-series
  chart always does, so identity is never carried by colour alone;
* grid and spines are recessive so the data is the darkest thing on the canvas.

The two categorical colours (blue/orange) were checked for colour-vision
deficiency separation before use.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Dict, Optional, Sequence

import matplotlib
import numpy as np

# Always headless: the CLI runs on cluster nodes and Streamlit renders server-side.
matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402  (must follow the backend choice)
from matplotlib.figure import Figure  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

if TYPE_CHECKING:  # pragma: no cover
    import pandas as pd

    from psaurus_pcr.batch import BatchResult
    from psaurus_pcr.pipeline import AnalysisResult
    from psaurus_pcr.qc import QCStats

# --- palette ---------------------------------------------------------------
SURFACE = "#fcfcfb"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
TEXT_MUTED = "#8a8880"
SERIES_1 = "#2a78d6"   # blue    -- kept / in-library / the default single series
SERIES_2 = "#eb6834"   # orange  -- removed / unexpected
SERIES_3 = "#1baf7a"   # aqua
SERIES_4 = "#eda100"   # yellow
SERIES_5 = "#e87ba4"   # magenta
GRID = "#dedcd6"

#: Fixed colour order for the batch read-fate stack.  Assigned by category, in
#: this order, never cycled -- so a sample that is missing a category does not
#: repaint the others.
FATE_COLORS = (SERIES_1, SERIES_2, SERIES_3, SERIES_4, SERIES_5)


def _new_figure(width: float = 7.2, height: float = 4.2):
    fig, ax = plt.subplots(figsize=(width, height), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    return fig, ax


def _style(ax, title: str, xlabel: str, ylabel: str, grid_axis: str = "y") -> None:
    ax.set_title(title, color=TEXT_PRIMARY, fontsize=12, pad=12, loc="left")
    ax.set_xlabel(xlabel, color=TEXT_SECONDARY, fontsize=10)
    ax.set_ylabel(ylabel, color=TEXT_SECONDARY, fontsize=10)
    ax.tick_params(colors=TEXT_SECONDARY, labelsize=9, length=3)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.grid(axis=grid_axis, color=GRID, linewidth=0.8, alpha=0.7)
    ax.set_axisbelow(True)


def _empty(ax, message: str) -> None:
    ax.text(
        0.5, 0.5, message,
        transform=ax.transAxes, ha="center", va="center",
        color=TEXT_MUTED, fontsize=11,
    )
    ax.set_xticks([])
    ax.set_yticks([])


def _nbins(values: Sequence[float], cap: int = 60) -> int:
    n = len(values)
    if n < 2:
        return 1
    return max(10, min(cap, int(n ** 0.5)))


def _shared_bins(values: Sequence[float], cap: int = 60) -> np.ndarray:
    """Bin edges spanning ``values``, to be reused by every series on the axes.

    Letting matplotlib choose edges per call would give the before- and
    after-filter histograms different bin widths over different ranges, so
    their bar heights would not be comparable -- the filtered series would look
    far smaller than it is.  One set of edges for both fixes that.
    """
    lo, hi = float(min(values)), float(max(values))
    if hi <= lo:
        lo, hi = lo - 0.5, hi + 0.5
    return np.linspace(lo, hi, _nbins(values, cap) + 1)


def _stacked_hist(ax, bins, kept, removed, kept_label: str, removed_label: str) -> None:
    """Stacked histogram of a kept/removed partition (kept on the bottom).

    Segments are separated by a thin surface-coloured edge rather than by
    transparency, so neither series is obscured and the stack total is exactly
    the pre-filter distribution.
    """
    series, colors, labels = [], [], []
    if kept:
        series.append(list(kept)); colors.append(SERIES_1); labels.append(kept_label)
    if removed:
        series.append(list(removed)); colors.append(SERIES_2); labels.append(removed_label)
    if not series:
        return
    ax.hist(
        series, bins=bins, stacked=True, histtype="stepfilled",
        color=colors, label=labels, edgecolor=SURFACE, linewidth=0.8,
    )
    if len(series) > 1:
        # Explicit handles so the legend reads bottom-of-stack first,
        # matching how the bars are actually built up.
        ax.legend(
            handles=[Patch(facecolor=c, label=l) for c, l in zip(colors, labels)],
            frameon=False, fontsize=9, labelcolor=TEXT_SECONDARY,
        )


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------


def quality_distribution_figure(
    stats: "QCStats",
    threshold: Optional[float] = None,
    metric_label: str = "mean Phred",
) -> Figure:
    """Per-read quality distribution, split into kept and removed reads.

    The threshold line makes it obvious whether the cutoff is slicing through
    the bulk of the run (too aggressive for this chemistry) or trimming a
    separate low-quality tail (what you want).
    """
    fig, ax = _new_figure()
    if not stats or not stats.qualities_all:
        _empty(ax, "no reads")
        return fig

    bins = _shared_bins(stats.qualities_all)
    _stacked_hist(
        ax, bins,
        kept=stats.qualities_pass, removed=stats.qualities_fail,
        kept_label=f"kept (n={stats.passing_reads:,})",
        removed_label=f"removed (n={stats.failing_reads:,})",
    )
    if threshold is not None:
        ax.axvline(threshold, color=TEXT_SECONDARY, linestyle="--", linewidth=1.5)
        ax.annotate(
            f"Q{threshold:g} cutoff",
            xy=(threshold, ax.get_ylim()[1]),
            xytext=(4, -10), textcoords="offset points",
            color=TEXT_SECONDARY, fontsize=9, ha="left", va="top",
        )
    _style(ax, "Read quality distribution", f"read quality ({metric_label})", "reads")
    fig.tight_layout()
    return fig


def read_length_distribution_figure(stats: "QCStats") -> Figure:
    """Raw read-length distribution, split into kept and removed reads.

    A length shift between the two tells you the quality filter is removing a
    particular product (short adapter dimers, or long chimeras) rather than
    sampling the run uniformly.
    """
    fig, ax = _new_figure()
    if not stats or not stats.lengths_all:
        _empty(ax, "no reads")
        return fig

    bins = _shared_bins(stats.lengths_all)
    _stacked_hist(
        ax, bins,
        kept=stats.lengths_pass, removed=stats.lengths_fail,
        kept_label=f"kept (n={stats.passing_reads:,})",
        removed_label=f"removed (n={stats.failing_reads:,})",
    )
    _style(ax, "Read length distribution", "read length (bp)", "reads")
    fig.tight_layout()
    return fig


def insert_length_distribution_figure(insert_lengths: Sequence[int]) -> Figure:
    """Length distribution of the extracted inserts (one entry per counted read).

    A tight unimodal peak is what a clean defined-library PCR looks like;
    shoulders usually mean truncation products or missed internal flanks.
    """
    fig, ax = _new_figure()
    if not insert_lengths:
        _empty(ax, "no inserts extracted")
        return fig
    ax.hist(
        insert_lengths, bins=_shared_bins(insert_lengths), color=SERIES_1,
        edgecolor=SURFACE, linewidth=1.0,
    )
    _style(ax, "Extracted insert length distribution", "insert length (bp)", "reads")
    fig.tight_layout()
    return fig


def _short_label(sequence: str, rank: int, width: int = 26) -> str:
    if len(sequence) <= width:
        body = sequence
    else:
        head = width // 2 - 1
        tail = width - head - 1
        body = f"{sequence[:head]}…{sequence[-tail:]}"
    return f"{rank}. {body}"


def top_sequences_figure(unique_frame: "pd.DataFrame", top_n: int = 20) -> Figure:
    """Horizontal bar chart of the most abundant extracted sequences.

    Horizontal because the category labels are DNA strings; sorted descending
    with the top bar first.  When a library was supplied the bars are split by
    library membership, which makes contamination visible at a glance.
    """
    top_n = max(1, int(top_n))
    height = max(3.0, 0.32 * min(top_n, len(unique_frame)) + 1.6)
    fig, ax = _new_figure(height=height)
    if unique_frame is None or unique_frame.empty:
        _empty(ax, "no sequences extracted")
        return fig

    head = unique_frame.head(top_n)
    labels = [
        _short_label(seq, rank)
        for rank, seq in enumerate(head["sequence"], start=1)
    ]
    counts = list(head["count"])
    positions = list(range(len(counts)))[::-1]  # rank 1 at the top

    has_library = "in_library" in head.columns
    if has_library:
        flags = [bool(v) for v in head["in_library"]]
        colors = [SERIES_1 if f else SERIES_2 for f in flags]
    else:
        colors = [SERIES_1] * len(counts)

    ax.barh(positions, counts, color=colors, edgecolor=SURFACE, linewidth=1.0, height=0.72)
    ax.set_yticks(positions)
    ax.set_yticklabels(labels, fontsize=8, fontfamily="monospace", color=TEXT_SECONDARY)

    # Direct value labels: <= top_n bars, so labelling each is legible, not noise.
    span = max(counts) if counts else 1
    for pos, count in zip(positions, counts):
        ax.text(
            count + span * 0.01, pos, f"{int(count):,}",
            va="center", ha="left", fontsize=8, color=TEXT_SECONDARY,
        )
    ax.set_xlim(0, span * 1.12)

    _style(
        ax,
        f"Top {len(counts)} most abundant extracted sequences",
        "reads",
        "",
        grid_axis="x",
    )
    if has_library and len(set(flags)) > 1:
        ax.legend(
            handles=[
                Patch(facecolor=SERIES_1, label="in library"),
                Patch(facecolor=SERIES_2, label="not in library"),
            ],
            frameon=False, fontsize=9, labelcolor=TEXT_SECONDARY, loc="lower right",
        )
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# bulk save
# ---------------------------------------------------------------------------


def build_all_figures(result: "AnalysisResult", top_n: int = 20) -> Dict[str, Figure]:
    """All four QC figures, keyed by output-file stem."""
    metric = result.run_summary["parameters"]["quality_metric"]
    metric_label = "mean Phred" if metric == "mean_phred" else "qscore from mean error prob."
    threshold = result.run_summary["parameters"]["min_mean_quality"]
    return {
        "quality_distribution": quality_distribution_figure(
            result.qc_stats, threshold, metric_label
        ),
        "read_length_distribution": read_length_distribution_figure(result.qc_stats),
        "insert_length_distribution": insert_length_distribution_figure(
            result.insert_lengths
        ),
        "top_sequences": top_sequences_figure(result.unique_sequences, top_n),
    }


def save_all_plots(
    result: "AnalysisResult", output_dir: Path | str, top_n: int = 20
) -> Dict[str, Path]:
    """Render and write every figure as PNG; returns label -> path."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    written: Dict[str, Path] = {}
    for stem, fig in build_all_figures(result, top_n=top_n).items():
        path = out / f"{stem}.png"
        fig.savefig(path, facecolor=fig.get_facecolor(), bbox_inches="tight")
        plt.close(fig)
        written[f"plot_{stem}"] = path
    return written


# ---------------------------------------------------------------------------
# batch (cross-sample) figures
# ---------------------------------------------------------------------------


def batch_read_fate_figure(batch: "BatchResult") -> Figure:
    """100% stacked bar of read fates per sample.

    Normalised to percent because samples in a batch rarely have comparable
    depth, and the question being asked is "did this sample behave like the
    others?", not "how deep was it?".  Absolute depth is kept on the tick
    labels so it is never lost.
    """
    from psaurus_pcr.batch import FATE_GROUPS, fate_breakdown

    frame = fate_breakdown(batch.results)
    height = max(3.0, 0.42 * len(frame) + 1.8)
    fig, ax = _new_figure(height=height)
    if frame.empty:
        _empty(ax, "no samples analysed")
        return fig

    labels = list(FATE_GROUPS.keys())
    positions = list(range(len(frame)))[::-1]  # first sample at the top
    left = [0.0] * len(frame)
    drawn = []
    for label, color in zip(labels, FATE_COLORS):
        values = [float(v) for v in frame[label]]
        if all(v == 0 for v in values):
            continue  # do not put an all-zero category in the legend
        ax.barh(positions, values, left=left, color=color, height=0.68,
                edgecolor=SURFACE, linewidth=1.0)
        drawn.append((label, color))
        left = [a + b for a, b in zip(left, values)]

    if not drawn:
        # Every sample had zero quality-passing reads: there is no composition
        # to draw, and an empty legend makes matplotlib raise.
        _empty(ax, "no quality-passing reads in any sample")
        _style(ax, "Read fate by sample", "% of quality-passing reads", "", grid_axis="x")
        fig.tight_layout()
        return fig

    ax.set_yticks(positions)
    ax.set_yticklabels(
        [f"{row.sample}  ({int(row.reads):,} reads)" for row in frame.itertuples()],
        fontsize=9, color=TEXT_SECONDARY,
    )
    # Direct label on the extracted segment: the one number people look for.
    for pos, value in zip(positions, frame[labels[0]]):
        if value >= 12:
            ax.text(value / 2, pos, f"{value:.0f}%", va="center", ha="center",
                    fontsize=8, color="#ffffff")
    ax.set_xlim(0, 100)
    _style(ax, "Read fate by sample", "% of quality-passing reads", "", grid_axis="x")
    ax.legend(
        handles=[Patch(facecolor=c, label=l) for l, c in drawn],
        frameon=False, fontsize=9, labelcolor=TEXT_SECONDARY,
        loc="upper center",
        # Offset scaled by figure height so the legend clears the x-label
        # whether the batch has 2 samples or 40.
        bbox_to_anchor=(0.5, -1.05 / height),
        ncol=min(3, len(drawn)),
    )
    fig.tight_layout()
    return fig


def batch_library_detection_figure(batch: "BatchResult") -> Optional[Figure]:
    """Percentage of the intended library detected in each sample.

    Returns ``None`` when the batch was run without a library, so the caller
    can simply skip the figure.
    """
    overview = batch.sample_overview
    if overview.empty or "percent_library_detected" not in overview.columns:
        return None
    frame = overview[overview["status"] == "ok"]
    if frame.empty:
        return None

    height = max(3.0, 0.38 * len(frame) + 1.6)
    fig, ax = _new_figure(height=height)
    positions = list(range(len(frame)))[::-1]
    values = [float(v) for v in frame["percent_library_detected"]]
    ax.barh(positions, values, color=SERIES_1, edgecolor=SURFACE, linewidth=1.0,
            height=0.68)
    ax.set_yticks(positions)
    ax.set_yticklabels(list(frame["sample"]), fontsize=9, color=TEXT_SECONDARY)
    for pos, value in zip(positions, values):
        ax.text(value + 1.5, pos, f"{value:.0f}%", va="center", ha="left",
                fontsize=8, color=TEXT_SECONDARY)
    ax.set_xlim(0, 105)
    _style(ax, "Intended library detected per sample",
           "% of library members detected", "", grid_axis="x")
    fig.tight_layout()
    return fig


def build_batch_figures(batch: "BatchResult") -> Dict[str, Figure]:
    """Cross-sample figures, keyed by output-file stem."""
    figures = {"batch_read_fate": batch_read_fate_figure(batch)}
    library_figure = batch_library_detection_figure(batch)
    if library_figure is not None:
        figures["batch_library_detection"] = library_figure
    return figures


def save_batch_plots(batch: "BatchResult", output_dir: Path | str) -> Dict[str, Path]:
    """Render and write the cross-sample figures; returns label -> path."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    written: Dict[str, Path] = {}
    for stem, fig in build_batch_figures(batch).items():
        path = out / f"{stem}.png"
        fig.savefig(path, facecolor=fig.get_facecolor(), bbox_inches="tight")
        plt.close(fig)
        written[f"plot_{stem}"] = path
    return written
