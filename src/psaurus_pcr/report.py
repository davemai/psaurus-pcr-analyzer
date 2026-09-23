"""Self-contained PDF run report, for record keeping.

Everything a run produced -- parameters, read accounting, the count tables,
the library comparison, every plot and, for a batch, the cross-sample views --
rendered into one archival PDF.  The point is that the PDF alone is enough to
reconstruct what was run and what came out, months later, without the TSVs.

Built on matplotlib's ``PdfPages`` so it adds no dependency beyond what the
plots already need, and so the figures in the report are the *same* figures the
rest of the tool produces rather than a second implementation of them.

Tables are rendered as monospace text: a run report is read, not manipulated,
and fixed-width text paginates predictably and stays legible at 6.5pt.  Very
long tables are truncated with an explicit note pointing at the TSV, so the
PDF never silently implies it is complete.
"""

from __future__ import annotations

import io
import os
from typing import TYPE_CHECKING, Iterable, List, Optional, Sequence, Union

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

from psaurus_pcr.plots import (  # noqa: E402
    SURFACE,
    TEXT_MUTED,
    TEXT_PRIMARY,
    TEXT_SECONDARY,
    build_all_figures,
    build_batch_figures,
)

if TYPE_CHECKING:  # pragma: no cover
    from psaurus_pcr.batch import BatchResult
    from psaurus_pcr.pipeline import AnalysisResult

#: A4 portrait, in inches.
PAGE_SIZE = (8.27, 11.69)

MARGIN_LEFT = 0.06
MARGIN_RIGHT = 0.94
TOP = 0.945
BOTTOM = 0.055

BODY_SIZE = 8.0
TABLE_SIZE = 6.4
LINESPACING = 1.35
#: Characters that fit across the page at :data:`TABLE_SIZE` in monospace.
TABLE_WIDTH = 128
#: Table rows per page, leaving room for the header and footer.
ROWS_PER_PAGE = 72

#: The overview columns worth printing, in reading order.  ``sample_overview``
#: carries ~25 columns (including a count for every read status); dumping them
#: all would spread four samples over eight near-empty pages.  The report shows
#: the ones you actually scan, plus a read-fate table of its own, and the TSV
#: keeps everything.
OVERVIEW_COLUMNS = [
    "sample", "status", "total_reads", "percent_passing_quality",
    "reads_extracted", "percent_extracted_of_qc", "unique_sequences",
    "singletons", "percent_singletons", "median_insert_length",
    "percent_reads_in_top_sequence", "reads_with_ambiguous_flank",
    "r_convention",
]
OVERVIEW_LIBRARY_COLUMNS = [
    "library_sequences_detected", "percent_library_detected",
    "unexpected_unique_sequences", "percent_reads_unexpected",
]

#: Cap on rows taken from any one table, so a 10k-sequence run does not
#: produce a 200-page PDF.  The note under a truncated table says so.
DEFAULT_MAX_TABLE_ROWS = 60


# ---------------------------------------------------------------------------
# page primitives
# ---------------------------------------------------------------------------


class _Report:
    """Accumulates pages and keeps the footer numbering consistent."""

    def __init__(self, pdf: PdfPages, label: str) -> None:
        self.pdf = pdf
        self.label = label
        self.page_number = 0

    def new_page(self) -> Figure:
        fig = plt.figure(figsize=PAGE_SIZE)
        fig.patch.set_facecolor(SURFACE)
        return fig

    def finish(self, fig: Figure) -> None:
        self.page_number += 1
        fig.text(
            MARGIN_LEFT, 0.028, self.label,
            fontsize=6.5, color=TEXT_MUTED, ha="left", va="center",
        )
        fig.text(
            MARGIN_RIGHT, 0.028, f"page {self.page_number}",
            fontsize=6.5, color=TEXT_MUTED, ha="right", va="center",
        )
        self.pdf.savefig(fig, facecolor=fig.get_facecolor())
        plt.close(fig)

    # -- building blocks ---------------------------------------------------

    def heading(self, fig: Figure, title: str, subtitle: str = "") -> float:
        """Draw a page heading; returns the y to start the body at."""
        fig.text(MARGIN_LEFT, TOP, title, fontsize=15, color=TEXT_PRIMARY,
                 ha="left", va="top", weight="bold")
        y = TOP - 0.028
        if subtitle:
            fig.text(MARGIN_LEFT, y, subtitle, fontsize=9, color=TEXT_SECONDARY,
                     ha="left", va="top")
            y -= 0.020
        fig.add_artist(
            plt.Line2D([MARGIN_LEFT, MARGIN_RIGHT], [y - 0.004, y - 0.004],
                       transform=fig.transFigure, color="#dedcd6", linewidth=1.0)
        )
        return y - 0.024

    def section(self, fig: Figure, title: str, y: float) -> float:
        fig.text(MARGIN_LEFT, y, title, fontsize=10.5, color=TEXT_PRIMARY,
                 ha="left", va="top", weight="bold")
        return y - 0.022

    def lines(self, fig: Figure, lines: Sequence[str], y: float,
              size: float = BODY_SIZE, color: str = TEXT_SECONDARY) -> float:
        """Draw monospace lines from ``y`` downwards; returns the new y.

        The advance must match what matplotlib actually draws: one line
        occupies ``fontsize * linespacing`` points, which is
        ``that / (page_height_inches * 72)`` in figure coordinates.  Getting
        this wrong is what leaves a report full of ragged gaps.
        """
        if not lines:
            return y
        line_height = size * LINESPACING / (PAGE_SIZE[1] * 72)
        fig.text(
            MARGIN_LEFT, y, "\n".join(lines),
            fontsize=size, color=color, ha="left", va="top",
            family="monospace", linespacing=LINESPACING,
        )
        return y - line_height * len(lines) - 0.008


# ---------------------------------------------------------------------------
# text / table formatting
# ---------------------------------------------------------------------------


def _shorten(value: str, width: int) -> str:
    if len(value) <= width:
        return value
    head = width // 2 - 1
    tail = width - head - 1
    return f"{value[:head]}…{value[-tail:]}"


def _cell(value, width: int) -> str:
    # Transposing a mixed-dtype frame yields numpy scalars in object columns;
    # unwrap them so ints keep their thousands separators and floats their
    # formatting instead of falling through to str().
    if hasattr(value, "item") and getattr(value, "shape", None) == ():
        try:
            value = value.item()
        except (ValueError, AttributeError):  # pragma: no cover - defensive
            pass
    if value is None or (isinstance(value, float) and pd.isna(value)):
        text = "-"
    elif isinstance(value, bool):
        text = "yes" if value else "no"
    elif isinstance(value, float):
        text = f"{value:,.2f}" if abs(value) >= 0.01 or value == 0 else f"{value:.3g}"
    elif isinstance(value, (int,)) and not isinstance(value, bool):
        text = f"{value:,}"
    else:
        text = str(value)
    return _shorten(text, width)


def _wrap_note(note: str, width: int = TABLE_WIDTH) -> List[str]:
    """Wrap a parenthetical note so it never runs off the page edge."""
    words = note.split()
    lines: List[str] = []
    current = ""
    for word in words:
        candidate = f"{current} {word}".strip()
        if len(candidate) > width and current:
            lines.append(current)
            current = word
        else:
            current = candidate
    if current:
        lines.append(current)
    return lines


def _render_block(frame: pd.DataFrame, columns: Sequence[str], rendered: dict,
                  widths: dict, total_width: int) -> List[str]:
    def row_text(cells: Iterable[str]) -> str:
        return "  ".join(
            cell.ljust(widths[column]) for column, cell in zip(columns, cells)
        ).rstrip()

    used = sum(widths[c] for c in columns) + 2 * (len(columns) - 1)
    lines = [row_text([str(c)[: widths[c]] for c in columns])]
    lines.append("-" * min(total_width, used))
    for index in range(len(frame)):
        lines.append(row_text([rendered[column][index] for column in columns]))
    return lines


def table_blocks(
    frame: pd.DataFrame,
    max_rows: Optional[int] = DEFAULT_MAX_TABLE_ROWS,
    max_cell_width: int = 30,
    total_width: int = TABLE_WIDTH,
    key_columns: int = 1,
) -> List[List[str]]:
    """Render a DataFrame as one or more fixed-width monospace blocks.

    A table wider than the page is **split across blocks by column group**
    rather than having its extra columns dropped: the first ``key_columns``
    are repeated in every block so each one can be read on its own.  For a
    record-keeping report, losing columns silently would be the worst possible
    behaviour -- the whole point is that the PDF is complete.

    Rows beyond ``max_rows`` are dropped, which *is* announced in a note.
    """
    if frame is None or frame.empty:
        return [["(no rows)"]]

    body = frame if max_rows is None else frame.head(max_rows)
    columns = [str(c) for c in body.columns]
    rendered = {
        str(column): [_cell(value, max_cell_width) for value in body[column]]
        for column in body.columns
    }
    widths = {
        column: min(
            max_cell_width,
            max([len(column)] + [len(cell) for cell in rendered[column]]),
        )
        for column in columns
    }

    keys = columns[: max(0, key_columns)]
    key_width = sum(widths[c] + 2 for c in keys)
    rest = columns[len(keys):]

    groups: List[List[str]] = []
    current: List[str] = []
    used = key_width
    for column in rest:
        need = widths[column] + 2
        if current and used + need > total_width:
            groups.append(current)
            current = []
            used = key_width
        current.append(column)
        used += need
    if current or not groups:
        groups.append(current)

    truncated = max_rows is not None and len(frame) > max_rows
    blocks: List[List[str]] = []
    for index, group in enumerate(groups):
        block_columns = keys + group
        lines = _render_block(body, block_columns, rendered, widths, total_width)
        notes = []
        if truncated:
            notes.append(
                f"showing {len(body)} of {len(frame):,} rows - "
                "full table in the TSV output"
            )
        if len(groups) > 1:
            notes.append(
                f"columns {index + 1} of {len(groups)}"
                + (f"; first {len(keys)} column(s) repeated" if keys else "")
            )
        for note in notes:
            lines.append("")
            lines.extend(f"({line})" for line in _wrap_note(note))
        blocks.append(lines)
    return blocks


def format_table(
    frame: pd.DataFrame,
    max_rows: Optional[int] = DEFAULT_MAX_TABLE_ROWS,
    max_cell_width: int = 30,
    total_width: int = TABLE_WIDTH,
) -> List[str]:
    """Single-block rendering, for tables known to fit the page width."""
    blocks = table_blocks(frame, max_rows, max_cell_width, total_width)
    if len(blocks) == 1:
        return blocks[0]
    lines: List[str] = []
    for block in blocks:
        if lines:
            lines.append("")
        lines.extend(block)
    return lines


def _paginate(report: _Report, title: str, subtitle: str, lines: Sequence[str]) -> None:
    """Split pre-rendered lines across pages, repeating the column header."""
    chunks = [lines[i : i + ROWS_PER_PAGE] for i in range(0, len(lines), ROWS_PER_PAGE)] or [[]]
    header_lines = list(lines[:2]) if len(lines) >= 2 else []
    for index, chunk in enumerate(chunks):
        fig = report.new_page()
        page_subtitle = subtitle if index == 0 else f"{subtitle} (continued)"
        y = report.heading(fig, title, page_subtitle)
        # Repeat the column header on continuation pages so a long table stays
        # readable away from its first page.
        block = list(chunk) if index == 0 else header_lines + list(chunk)
        report.lines(fig, block, y, size=TABLE_SIZE)
        report.finish(fig)


def _table_pages(report: _Report, title: str, subtitle: str, lines: Sequence[str]) -> None:
    _paginate(report, title, subtitle, lines)


def _select(frame: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    """Subset to ``columns``, keeping only the ones the frame actually has."""
    present = [c for c in columns if c in frame.columns]
    return frame[present] if present else frame


#: Above this many samples, a per-sample table is left as one row per sample.
#: Below it, transposing (metrics down the side, samples across) is far more
#: compact: four samples with twenty metrics is one readable page transposed
#: versus four near-empty pages of column groups.
TRANSPOSE_SAMPLE_LIMIT = 12


def _orient_for_report(frame: pd.DataFrame, key: str = "sample") -> pd.DataFrame:
    """Transpose a narrow-and-wide per-sample table so it fits the page."""
    if frame.empty or key not in frame.columns or len(frame) > TRANSPOSE_SAMPLE_LIMIT:
        return frame
    transposed = frame.set_index(key).T
    transposed.index.name = "metric"
    transposed.columns.name = None
    return transposed.reset_index()


def _frame_pages(
    report: _Report,
    title: str,
    subtitle: str,
    frame: pd.DataFrame,
    max_rows: Optional[int] = DEFAULT_MAX_TABLE_ROWS,
    key_columns: int = 1,
    max_cell_width: int = 30,
) -> None:
    """Paginate a DataFrame, splitting by columns first, then by rows."""
    blocks = table_blocks(frame, max_rows=max_rows, key_columns=key_columns,
                          max_cell_width=max_cell_width)
    for index, block in enumerate(blocks):
        block_subtitle = (
            subtitle if len(blocks) == 1
            else f"{subtitle} - columns {index + 1}/{len(blocks)}"
        )
        _paginate(report, title, block_subtitle, block)


# ---------------------------------------------------------------------------
# figures on a page
# ---------------------------------------------------------------------------


def _figure_image(fig: Figure, dpi: int = 200):
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=dpi, facecolor=fig.get_facecolor(),
                bbox_inches="tight")
    buffer.seek(0)
    return plt.imread(buffer)


def _place_image(page: Figure, image, rect) -> None:
    """Draw ``image`` inside ``rect`` (figure coords), preserving aspect."""
    left, bottom, width, height = rect
    image_aspect = image.shape[1] / image.shape[0]
    # Convert the box to inches so the comparison is in real units.
    box_w_in, box_h_in = width * PAGE_SIZE[0], height * PAGE_SIZE[1]
    if box_w_in / box_h_in > image_aspect:
        draw_h_in = box_h_in
        draw_w_in = draw_h_in * image_aspect
    else:
        draw_w_in = box_w_in
        draw_h_in = draw_w_in / image_aspect
    draw_w = draw_w_in / PAGE_SIZE[0]
    draw_h = draw_h_in / PAGE_SIZE[1]
    axes = page.add_axes([
        left + (width - draw_w) / 2,
        bottom + (height - draw_h) / 2,
        draw_w,
        draw_h,
    ])
    axes.imshow(image)
    axes.axis("off")


#: A figure at least this much wider than tall can share a page; anything
#: squarer or taller (the top-N bar chart, a long batch read-fate stack) gets
#: a page to itself, because letterboxing it into half a page renders its
#: axis labels unreadably small.
PAIRABLE_ASPECT = 1.35


def _plot_pages(report: _Report, title: str, subtitle: str, figures: Sequence[Figure]) -> None:
    """Lay figures out one or two per page, depending on their shape."""
    images = [_figure_image(fig) for fig in figures]
    for fig in figures:
        plt.close(fig)

    # Greedily pair consecutive wide figures; tall ones stand alone.
    groups: List[List] = []
    pending: List = []
    for image in images:
        if image.shape[1] / image.shape[0] >= PAIRABLE_ASPECT:
            pending.append(image)
            if len(pending) == 2:
                groups.append(pending)
                pending = []
        else:
            if pending:
                groups.append(pending)
                pending = []
            groups.append([image])
    if pending:
        groups.append(pending)

    full_width = MARGIN_RIGHT - MARGIN_LEFT
    for index, group in enumerate(groups):
        page = report.new_page()
        page_subtitle = subtitle if index == 0 else f"{subtitle} (continued)"
        y = report.heading(page, title, page_subtitle)
        available = y - BOTTOM - 0.02
        if len(group) == 2:
            half = available / 2
            _place_image(page, group[0], (MARGIN_LEFT, y - half, full_width, half))
            _place_image(page, group[1], (MARGIN_LEFT, BOTTOM + 0.01, full_width, half))
        else:
            _place_image(page, group[0], (MARGIN_LEFT, BOTTOM + 0.01, full_width, available))
        report.finish(page)


# ---------------------------------------------------------------------------
# content builders
# ---------------------------------------------------------------------------


def _kv_lines(pairs: Sequence[tuple], width: int = 30) -> List[str]:
    return [f"{str(key) + ' ':.<{width}} {value}" for key, value in pairs]


def _parameter_pairs(parameters: dict) -> List[tuple]:
    """The parameters worth reading back, in a sensible order."""
    order = [
        ("min_mean_quality", "Minimum mean read quality"),
        ("quality_metric", "Quality statistic"),
        ("flank_edit_fraction", "Flank edit fraction"),
        ("forward_max_edits", "Forward max edits (explicit)"),
        ("reverse_max_edits", "Reverse max edits (explicit)"),
        ("forward_max_edits_used", "Forward edit budget used"),
        ("reverse_max_edits_used", "Reverse edit budget used"),
        ("ambiguity_margin", "Ambiguity margin"),
        ("exclude_ambiguous", "Ambiguous reads excluded"),
        ("r_convention", "R convention requested"),
        ("orientation_probe_reads", "Orientation probe reads"),
        ("max_insert_length", "Max insert length (explicit)"),
        ("max_insert_length_used", "Max insert length used"),
        ("min_insert_length", "Min insert length"),
        ("library_length_multiplier", "Library length multiplier"),
        ("fuzzy_library", "Fuzzy library matching"),
        ("library_edit_fraction", "Library edit fraction"),
        ("library_max_edits", "Library max edits"),
    ]
    pairs = []
    for key, label in order:
        if key not in parameters:
            continue
        value = parameters[key]
        if value is None:
            value = "(derived)" if key.endswith("max_edits") or "max_insert" in key else "-"
        elif isinstance(value, bool):
            value = "yes" if value else "no"
        pairs.append((label, value))
    return pairs


def _cover_page(report: _Report, title: str, subtitle: str, blocks: Sequence[tuple]) -> None:
    fig = report.new_page()
    y = report.heading(fig, title, subtitle)
    y -= 0.008
    for block_title, pairs in blocks:
        y = report.section(fig, block_title, y)
        y = report.lines(fig, _kv_lines(pairs), y)
        y -= 0.010
    report.finish(fig)


def _sample_summary_pairs(result: "AnalysisResult") -> List[tuple]:
    summary = result.run_summary
    quality = summary["quality"]
    extraction = summary["extraction"]
    quant = summary["quantification"]
    return [
        ("Total reads", f"{quality['total_reads']:,}"),
        ("Passed quality filter",
         f"{quality['reads_passing_quality_filter']:,} "
         f"({quality['percent_passing_quality_filter']:.2f}%)"),
        ("Inserts extracted",
         f"{quant['total_extracted_reads']:,} "
         f"({extraction['status_percent'].get('extracted', 0.0):.2f}% of QC-passing)"),
        ("Unique sequences", f"{quant['unique_sequences']:,}"),
        ("Singletons",
         f"{quant['singletons']:,} ({quant['percent_singletons']:.2f}% of unique)"),
        ("Median insert length", quant["insert_length"]["median"]),
        ("Reads with ambiguous flank", f"{extraction['reads_with_ambiguous_flank']:,}"),
        ("R convention resolved", summary["inputs"]["r_convention"]["resolved"]),
    ]


def _status_frame(result: "AnalysisResult") -> pd.DataFrame:
    extraction = result.run_summary["extraction"]
    return pd.DataFrame([
        {
            "status": status,
            "reads": count,
            "percent_of_qc_passing": extraction["status_percent"].get(status, 0.0),
        }
        for status, count in extraction["status_counts"].items()
    ])


def _add_sample_sections(
    report: _Report,
    name: str,
    result: "AnalysisResult",
    top_n: int,
    max_table_rows: int,
) -> None:
    summary = result.run_summary
    label = f"Sample: {name}"

    # --- QC page -----------------------------------------------------------
    fig = report.new_page()
    y = report.heading(fig, label, "Quality control and read accounting")
    y = report.section(fig, "Summary", y)
    y = report.lines(fig, _kv_lines(_sample_summary_pairs(result)), y)
    y -= 0.008
    y = report.section(fig, "Read fate", y)
    y = report.lines(fig, format_table(_status_frame(result), max_rows=None), y,
                     size=TABLE_SIZE)
    y -= 0.008
    y = report.section(fig, "Read length and quality", y)
    quality = summary["quality"]
    stats_frame = pd.DataFrame([
        {"distribution": key.replace("_", " "), **value}
        for key, value in (
            ("read length before filter", quality["read_length_before_filter"]),
            ("read length after filter", quality["read_length_after_filter"]),
            ("read quality before filter", quality["read_quality_before_filter"]),
            ("read quality after filter", quality["read_quality_after_filter"]),
            ("extracted insert length", summary["quantification"]["insert_length"]),
        )
    ])
    y = report.lines(fig, format_table(stats_frame, max_rows=None), y, size=TABLE_SIZE)
    y -= 0.008
    probe = summary["inputs"]["r_convention"]
    y = report.section(fig, "R orientation", y)
    report.lines(fig, _kv_lines([
        ("Requested", probe["requested"]),
        ("Resolved", probe["resolved"]),
        ("Reads probed", probe["reads_probed"]),
        ("Extracted with literal R", probe["extracted_with_literal_R"]),
        ("Extracted with revcomp R", probe["extracted_with_revcomp_R"]),
        ("Sequence searched", summary["inputs"]["reverse_flank_searched"]),
        ("Note", _shorten(probe["note"], 70)),
    ]), y)
    report.finish(fig)

    # --- plots -------------------------------------------------------------
    figures = build_all_figures(result, top_n=top_n)
    _plot_pages(report, label, "Plots", list(figures.values()))

    # --- tables ------------------------------------------------------------
    _frame_pages(
        report, label, f"Most abundant extracted sequences (top {top_n})",
        result.unique_sequences.head(top_n), max_rows=None,
    )
    if result.library_comparison is not None:
        lib = summary["library"]
        fig = report.new_page()
        y = report.heading(fig, label, "Intended library comparison")
        y = report.section(fig, "Summary", y)
        y = report.lines(fig, _kv_lines([
            ("Matching mode", lib["matching_mode"]),
            ("Library members", f"{lib['library_sequences_unique']:,}"),
            ("Detected",
             f"{lib['library_sequences_detected']:,} ({lib['percent_library_detected']:.2f}%)"),
            ("Not detected", f"{lib['library_sequences_not_detected']:,}"),
            ("Unexpected unique sequences",
             f"{lib['unexpected_unique_sequences']:,} "
             f"({lib['percent_unique_sequences_unexpected']:.2f}%)"),
            ("Reads in unexpected sequences",
             f"{lib['reads_in_unexpected_sequences']:,} "
             f"({lib['percent_reads_unexpected']:.2f}%)"),
        ]), y)
        y -= 0.008
        report.finish(fig)
        _frame_pages(report, label, "Intended library comparison - per member",
                     result.library_comparison, max_rows=max_table_rows)


# ---------------------------------------------------------------------------
# public API
# ---------------------------------------------------------------------------


def write_sample_report(
    result: "AnalysisResult",
    target: Union[str, os.PathLike, io.IOBase],
    top_n: int = 20,
    max_table_rows: int = DEFAULT_MAX_TABLE_ROWS,
) -> None:
    """Write a one-sample PDF report to ``target`` (a path or a binary stream)."""
    summary = result.run_summary
    with PdfPages(target) as pdf:
        report = _Report(pdf, f"psaurus-pcr {summary['version']} - {summary['timestamp_local']}")
        _cover_page(
            report,
            "Plasmidsaurus PCR amplicon analysis",
            f"Run report - {summary['timestamp_local']}",
            [
                ("Inputs", [
                    ("FASTQ", _shorten(str(summary["inputs"]["fastq"]), 60)),
                    ("Library", _shorten(str(summary["inputs"]["library"] or "(none)"), 60)),
                    ("Forward flank (F)", summary["inputs"]["forward_flank"]),
                    ("Reverse flank (R), as given", summary["inputs"]["reverse_flank_as_given"]),
                    ("Reverse flank, searched", summary["inputs"]["reverse_flank_searched"]),
                    ("R convention", summary["inputs"]["r_convention"]["resolved"]),
                ]),
                ("Results", _sample_summary_pairs(result)),
                ("Parameters", _parameter_pairs(summary["parameters"])),
                ("Run", [
                    ("Tool version", summary["version"]),
                    ("Timestamp (local)", summary["timestamp_local"]),
                    ("Timestamp (UTC)", summary["timestamp_utc"]),
                    ("Runtime (s)", summary["runtime_seconds"]),
                ]),
            ],
        )
        fastq_label = str(summary["inputs"]["fastq"])
        _add_sample_sections(
            report, os.path.basename(fastq_label) or fastq_label,
            result, top_n, max_table_rows,
        )


def write_batch_report(
    batch: "BatchResult",
    target: Union[str, os.PathLike, io.IOBase],
    top_n: int = 20,
    max_table_rows: int = DEFAULT_MAX_TABLE_ROWS,
    per_sample_detail: bool = True,
    max_detailed_samples: Optional[int] = 24,
) -> None:
    """Write a batch PDF report: overview first, then a section per sample.

    ``max_detailed_samples`` caps how many samples get their own pages, so a
    96-barcode plate produces a usable document rather than a 500-page one;
    the overview and every combined table still cover all of them, and the cap
    is stated in the report.
    """
    info = batch.batch_summary
    with PdfPages(target) as pdf:
        report = _Report(pdf, f"psaurus-pcr batch - {info['timestamp_local']}")

        sample_lines = [
            f"{item['sample']}  <-  {item.get('container') or ''}{item['origin']}"
            for item in info["inputs"]["samples"]
        ]
        _cover_page(
            report,
            "Plasmidsaurus PCR amplicon analysis",
            f"Batch report - {info['samples_analysed']} sample(s) "
            f"- {info['timestamp_local']}",
            [
                ("Inputs", [
                    ("Samples discovered", info["samples_discovered"]),
                    ("Samples analysed", info["samples_analysed"]),
                    ("Samples failed", info["samples_failed"]),
                    ("Library", _shorten(str(info["inputs"]["library"] or "(none)"), 60)),
                    ("Library sequences", info["inputs"]["library_sequences"]),
                    ("Forward flank (F)", info["inputs"]["forward_flank"]),
                    ("Reverse flank (R), as given", info["inputs"]["reverse_flank_as_given"]),
                ]),
                ("Totals", [
                    ("Total reads", f"{info['totals']['total_reads']:,}"),
                    ("Passed quality filter", f"{info['totals']['reads_passing_quality']:,}"),
                    ("Inserts extracted", f"{info['totals']['reads_extracted']:,}"),
                    ("Unique sequences across batch",
                     f"{info['totals']['unique_sequences_across_batch']:,}"),
                ]),
                ("Parameters", _parameter_pairs(info["parameters"])),
                ("Run", [
                    ("Tool version", info.get("version", "")),
                    ("Timestamp (local)", info["timestamp_local"]),
                    ("Timestamp (UTC)", info["timestamp_utc"]),
                    ("Runtime (s)", info["runtime_seconds"]),
                ]),
            ],
        )

        _table_pages(report, "Batch", "Samples analysed",
                     sample_lines or ["(none)"])
        if batch.failures:
            _frame_pages(
                report, "Batch", "Samples that failed",
                pd.DataFrame([f.to_dict() for f in batch.failures]), max_rows=None,
            )

        overview = batch.sample_overview
        columns = list(OVERVIEW_COLUMNS)
        if "percent_library_detected" in overview.columns:
            columns += OVERVIEW_LIBRARY_COLUMNS
        if "error" in overview.columns and overview["error"].notna().any():
            columns.append("error")
        _frame_pages(report, "Batch overview", "Per-sample summary",
                     _orient_for_report(_select(overview, columns)),
                     max_rows=None, key_columns=1)

        fate_columns = ["sample"] + [
            c for c in overview.columns if str(c).startswith("reads_")
            and c != "reads_with_ambiguous_flank"
        ]
        _frame_pages(
            report, "Batch overview", "Read fate counts per sample",
            _orient_for_report(
                _select(overview, fate_columns).rename(
                    columns=lambda c: str(c)[len("reads_"):]
                    if str(c).startswith("reads_") else c
                )
            ),
            max_rows=None, key_columns=1,
        )
        _plot_pages(report, "Batch overview", "Cross-sample plots",
                    list(build_batch_figures(batch).values()))
        _frame_pages(
            report, "Batch overview", "Sequence x sample counts",
            batch.sequence_count_matrix, max_rows=max_table_rows,
        )
        if batch.library_detection_matrix is not None:
            _frame_pages(
                report, "Batch overview", "Library member x sample counts",
                batch.library_detection_matrix, max_rows=max_table_rows, key_columns=2,
            )

        if per_sample_detail:
            names = batch.sample_names
            shown = names if max_detailed_samples is None else names[:max_detailed_samples]
            for name in shown:
                _add_sample_sections(report, name, batch[name], top_n, max_table_rows)
            if len(shown) < len(names):
                fig = report.new_page()
                y = report.heading(fig, "Per-sample detail", "Truncated")
                report.lines(fig, [
                    f"Detailed pages were produced for the first {len(shown)} of "
                    f"{len(names)} samples.",
                    "",
                    "Every sample is still covered by the batch overview table and",
                    "the combined TSV outputs. Raise max_detailed_samples (or run",
                    "the samples in smaller batches) for full per-sample detail.",
                    "",
                    "Samples without detailed pages:",
                ] + [f"  {name}" for name in names[len(shown):]], y)
                report.finish(fig)


def report_bytes(
    result_or_batch, top_n: int = 20, **kwargs
) -> bytes:
    """Render a report to bytes -- what the Streamlit download button serves."""
    buffer = io.BytesIO()
    write_report(result_or_batch, buffer, top_n=top_n, **kwargs)
    return buffer.getvalue()


def write_report(result_or_batch, target, top_n: int = 20, **kwargs) -> None:
    """Dispatch to the sample or batch writer based on what it is given."""
    from psaurus_pcr.batch import BatchResult

    if isinstance(result_or_batch, BatchResult):
        write_batch_report(result_or_batch, target, top_n=top_n, **kwargs)
    else:
        kwargs.pop("per_sample_detail", None)
        kwargs.pop("max_detailed_samples", None)
        write_sample_report(result_or_batch, target, top_n=top_n, **kwargs)
