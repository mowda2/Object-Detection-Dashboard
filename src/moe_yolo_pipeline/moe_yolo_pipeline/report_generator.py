"""
Report Generator Module for TrafficIQ

Produces publication-quality exports from a :class:`~interval_binning.TrafficStudyResult`:

* **PDF** – multi-page turning movement count report suitable for a Traffic
  Impact Study appendix (cover page, data table in landscape, optional speed
  summary).
* **UTDF CSV** – Synchro-compatible ``Volume.csv`` for direct import into
  Synchro / SimTraffic.
* **Speed CSV** – per-vehicle speed log with a statistical summary footer.

Functions:
    generate_tmc_pdf          – PDF report via *reportlab*
    generate_utdf_csv         – Synchro UTDF Volume.csv
    generate_speed_summary_csv – speed data + 85th-percentile summary
"""

from __future__ import annotations

import csv
import io
import math
import os
import statistics
from collections import defaultdict
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .interval_binning import (
    IntervalBin,
    TrafficStudyResult,
    VEHICLE_CLASSES,
    _APPROACHES,
    _MOVEMENTS,
)

# ---------------------------------------------------------------------------
# Lazy reportlab imports (only needed for PDF generation)
# ---------------------------------------------------------------------------

def _import_reportlab():
    """Import reportlab modules on demand so the rest of the module works even
    when reportlab is not installed (e.g. for CSV-only usage)."""
    # Top-level
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
    from reportlab.lib.pagesizes import letter, landscape
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import inch, mm
    from reportlab.platypus import (
        BaseDocTemplate,
        Frame,
        NextPageTemplate,
        PageBreak,
        PageTemplate,
        Paragraph,
        SimpleDocTemplate,
        Spacer,
        Table,
        TableStyle,
    )
    from reportlab.graphics.shapes import Drawing, Line, Rect, String
    from reportlab.graphics.charts.barcharts import VerticalBarChart

    return {
        "colors": colors,
        "TA_CENTER": TA_CENTER,
        "TA_LEFT": TA_LEFT,
        "TA_RIGHT": TA_RIGHT,
        "letter": letter,
        "landscape": landscape,
        "ParagraphStyle": ParagraphStyle,
        "getSampleStyleSheet": getSampleStyleSheet,
        "inch": inch,
        "mm": mm,
        "BaseDocTemplate": BaseDocTemplate,
        "Frame": Frame,
        "NextPageTemplate": NextPageTemplate,
        "PageBreak": PageBreak,
        "PageTemplate": PageTemplate,
        "Paragraph": Paragraph,
        "SimpleDocTemplate": SimpleDocTemplate,
        "Spacer": Spacer,
        "Table": Table,
        "TableStyle": TableStyle,
        "Drawing": Drawing,
        "Line": Line,
        "Rect": Rect,
        "String": String,
        "VerticalBarChart": VerticalBarChart,
    }


# ---------------------------------------------------------------------------
# Colour palette (resolved lazily)
# ---------------------------------------------------------------------------

_HEADER_BG = None  # dark blue-grey
_SUBHEADER_BG = None  # lighter blue-grey
_PEAK_ROW_BG = None  # pale yellow highlight
_GRID_COLOR = None
_WHITE = None


def _init_colors():
    global _HEADER_BG, _SUBHEADER_BG, _PEAK_ROW_BG, _GRID_COLOR, _WHITE
    from reportlab.lib import colors
    _HEADER_BG = colors.HexColor("#2C3E50")
    _SUBHEADER_BG = colors.HexColor("#5D6D7E")
    _PEAK_ROW_BG = colors.HexColor("#FFF9C4")
    _GRID_COLOR = colors.HexColor("#BDC3C7")
    _WHITE = colors.white


# ===================================================================
# 1. PDF REPORT
# ===================================================================

def generate_tmc_pdf(
    result: TrafficStudyResult,
    output_path: str,
    weather: str = "Not recorded",
    speed_data: Optional[List[Dict[str, Any]]] = None,
) -> str:
    """Generate a professional PDF Turning Movement Count report.

    Parameters:
        result:      A :class:`~interval_binning.TrafficStudyResult`.
        output_path: Filesystem path for the generated PDF.
        weather:     Weather description for the cover page.
        speed_data:  Optional list of per-vehicle speed dicts (keys:
                     ``Track_ID``, ``Class``, ``Speed_KMH``, ``Timestamp_S``,
                     ``Approach``).  When provided, a speed summary page is
                     appended.

    Returns:
        The *output_path* string (for convenience chaining).
    """
    rl = _import_reportlab()
    _init_colors()

    inch = rl["inch"]
    landscape = rl["landscape"]
    letter = rl["letter"]

    # ---- Build story (list of Flowables) ----
    story: list = []

    # Page 1 — Cover / Summary (portrait)
    story.extend(_build_cover_page(result, weather, rl))
    story.append(rl["NextPageTemplate"]("landscape"))
    story.append(rl["PageBreak"]())

    # Page 2+ — TMC Data Table (landscape)
    story.extend(_build_tmc_table_pages(result, rl))

    # Last page — Speed Summary (landscape, optional)
    if speed_data:
        story.append(rl["PageBreak"]())
        story.extend(_build_speed_page(speed_data, rl))

    # ---- Build document with portrait + landscape templates ----
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    doc = rl["BaseDocTemplate"](
        output_path,
        pagesize=letter,
        leftMargin=0.6 * inch,
        rightMargin=0.6 * inch,
        topMargin=0.6 * inch,
        bottomMargin=0.6 * inch,
    )

    portrait_frame = rl["Frame"](
        doc.leftMargin, doc.bottomMargin,
        doc.width, doc.height,
        id="portrait",
    )
    landscape_w, landscape_h = landscape(letter)
    landscape_frame = rl["Frame"](
        0.5 * inch, 0.5 * inch,
        landscape_w - 1.0 * inch,
        landscape_h - 1.0 * inch,
        id="landscape",
    )

    doc.addPageTemplates([
        rl["PageTemplate"](id="portrait", frames=[portrait_frame],
                           pagesize=letter,
                           onPage=_footer_portrait),
        rl["PageTemplate"](id="landscape", frames=[landscape_frame],
                           pagesize=landscape(letter),
                           onPage=_footer_landscape),
    ])

    doc.build(story)
    return output_path


# ---------------------------------------------------------------------------
# Cover page
# ---------------------------------------------------------------------------

def _build_cover_page(result: TrafficStudyResult, weather: str, rl: dict) -> list:
    """Return Flowable elements for the cover/summary page."""
    styles = rl["getSampleStyleSheet"]()
    inch = rl["inch"]
    Paragraph = rl["Paragraph"]
    Spacer = rl["Spacer"]
    Table = rl["Table"]
    TableStyle = rl["TableStyle"]
    colors_mod = rl["colors"]

    elements: list = []

    # Title
    title_style = rl["ParagraphStyle"](
        "CoverTitle",
        parent=styles["Title"],
        fontSize=22,
        leading=28,
        textColor=_HEADER_BG,
        spaceAfter=6,
    )
    elements.append(Paragraph("Turning Movement Count", title_style))

    subtitle_style = rl["ParagraphStyle"](
        "CoverSubtitle",
        parent=styles["Title"],
        fontSize=16,
        leading=20,
        textColor=_SUBHEADER_BG,
        spaceAfter=24,
    )
    elements.append(Paragraph(result.intersection_name, subtitle_style))
    elements.append(Spacer(1, 0.15 * inch))

    # Info table
    info_data = [
        ["Study Date:", result.study_date],
        ["Study Period:", f"{result.study_start} – {result.study_end}"],
        ["Weather:", weather],
    ]
    info_table = Table(info_data, colWidths=[1.8 * inch, 4.5 * inch])
    info_table.setStyle(TableStyle([
        ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
        ("FONTNAME", (1, 0), (1, -1), "Helvetica"),
        ("FONTSIZE", (0, 0), (-1, -1), 11),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
    ]))
    elements.append(info_table)
    elements.append(Spacer(1, 0.4 * inch))

    # Summary heading
    heading_style = rl["ParagraphStyle"](
        "SummaryHeading",
        parent=styles["Heading2"],
        fontSize=14,
        textColor=_HEADER_BG,
        spaceAfter=10,
    )
    elements.append(Paragraph("Study Summary", heading_style))

    # Summary table
    summary_data = [
        ["Metric", "Value"],
        ["Total Volume", f"{result.total_volume:,}"],
        ["Heavy Vehicle %", f"{result.heavy_vehicle_pct:.1f}%"],
        ["AM Peak Hour", result.am_peak_hour],
        ["AM Peak Volume", f"{result.am_peak_volume:,}"],
        ["AM PHF", f"{result.am_peak_phf:.3f}"],
        ["PM Peak Hour", result.pm_peak_hour],
        ["PM Peak Volume", f"{result.pm_peak_volume:,}"],
        ["PM PHF", f"{result.pm_peak_phf:.3f}"],
    ]
    summary_table = Table(summary_data, colWidths=[2.5 * inch, 2.5 * inch])
    summary_table.setStyle(TableStyle([
        # Header row
        ("BACKGROUND", (0, 0), (-1, 0), _HEADER_BG),
        ("TEXTCOLOR", (0, 0), (-1, 0), _WHITE),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, 0), 11),
        ("ALIGN", (0, 0), (-1, 0), "CENTER"),
        # Body
        ("FONTNAME", (0, 1), (0, -1), "Helvetica-Bold"),
        ("FONTNAME", (1, 1), (1, -1), "Helvetica"),
        ("FONTSIZE", (0, 1), (-1, -1), 10),
        ("ALIGN", (1, 1), (1, -1), "CENTER"),
        # Grid
        ("GRID", (0, 0), (-1, -1), 0.5, _GRID_COLOR),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors_mod.white, colors_mod.HexColor("#F4F6F7")]),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]))
    elements.append(summary_table)
    elements.append(Spacer(1, 0.4 * inch))

    # Approach summary
    elements.append(Paragraph("Approach Summary (Vehicles)", heading_style))
    approach_data = _build_approach_summary_rows(result)
    approach_table = Table(approach_data, colWidths=[1.1 * inch] * 5)
    approach_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), _HEADER_BG),
        ("TEXTCOLOR", (0, 0), (-1, 0), _WHITE),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 10),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("GRID", (0, 0), (-1, -1), 0.5, _GRID_COLOR),
        ("FONTNAME", (0, 1), (0, -1), "Helvetica-Bold"),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    elements.append(approach_table)

    return elements


def _build_approach_summary_rows(result: TrafficStudyResult) -> list[list[str]]:
    """Aggregate bins into an approach × movement summary table."""
    totals: Dict[str, Dict[str, int]] = {
        ap: {mv: 0 for mv in ("L", "T", "R", "U")} for ap in _APPROACHES
    }
    for b in result.bins:
        totals[b.approach][b.movement] += b.total

    header = ["Approach", "Left", "Through", "Right", "U-Turn"]
    rows = [header]
    for ap in _APPROACHES:
        rows.append([
            ap,
            str(totals[ap]["L"]),
            str(totals[ap]["T"]),
            str(totals[ap]["R"]),
            str(totals[ap]["U"]),
        ])
    # Totals row
    rows.append([
        "Total",
        str(sum(totals[ap]["L"] for ap in _APPROACHES)),
        str(sum(totals[ap]["T"] for ap in _APPROACHES)),
        str(sum(totals[ap]["R"] for ap in _APPROACHES)),
        str(sum(totals[ap]["U"] for ap in _APPROACHES)),
    ])
    return rows


# ---------------------------------------------------------------------------
# TMC data table (landscape pages)
# ---------------------------------------------------------------------------

def _build_tmc_table_pages(result: TrafficStudyResult, rl: dict) -> list:
    """Build the main TMC data table as landscape-oriented Flowables."""
    inch = rl["inch"]
    Paragraph = rl["Paragraph"]
    Spacer = rl["Spacer"]
    Table = rl["Table"]
    TableStyle = rl["TableStyle"]
    colors_mod = rl["colors"]
    styles = rl["getSampleStyleSheet"]()

    elements: list = []

    heading_style = rl["ParagraphStyle"](
        "TMCHeading",
        parent=styles["Heading2"],
        fontSize=13,
        textColor=_HEADER_BG,
        spaceAfter=8,
    )
    elements.append(Paragraph("15-Minute Turning Movement Counts", heading_style))
    elements.append(Spacer(1, 0.1 * inch))

    # ---- Collect unique sorted intervals ----
    sorted_intervals = sorted(
        {b.interval_start for b in result.bins},
        key=lambda s: (int(s.split(":")[0]), int(s.split(":")[1])),
    )

    # ---- Determine peak-hour intervals for highlighting ----
    am_intervals = _parse_peak_intervals(result.am_peak_hour)
    pm_intervals = _parse_peak_intervals(result.pm_peak_hour)
    peak_intervals = am_intervals | pm_intervals

    # ---- Pre-aggregate: interval × approach × movement → total, heavy ----
    lookup: Dict[Tuple[str, str, str], Tuple[int, int]] = {}
    for b in result.bins:
        lookup[(b.interval_start, b.approach, b.movement)] = (
            b.total, b.heavy_vehicle_count,
        )

    # Pedestrian / Bicycle per interval × approach
    ped_lookup: Dict[Tuple[str, str], int] = defaultdict(int)
    bike_lookup: Dict[Tuple[str, str], int] = defaultdict(int)
    for b in result.bins:
        ped_lookup[(b.interval_start, b.approach)] = max(
            ped_lookup[(b.interval_start, b.approach)], b.pedestrian_count)
        bike_lookup[(b.interval_start, b.approach)] = max(
            bike_lookup[(b.interval_start, b.approach)], b.bicycle_count)

    # ---- Build header rows ----
    # Row 0: top-level groups
    # Columns: Time | NB(L T R) | SB(L T R) | EB(L T R) | WB(L T R) | Peds(NB SB EB WB) | Int Total
    movements = ["L", "T", "R"]
    num_mv = len(movements)

    header_row_0 = [""]
    for ap in _APPROACHES:
        header_row_0.append(ap)
        header_row_0.extend([""] * (num_mv - 1))
    header_row_0.append("Peds")
    header_row_0.extend([""] * (len(_APPROACHES) - 1))
    header_row_0.append("Int")

    header_row_1 = ["Time"]
    for _ap in _APPROACHES:
        header_row_1.extend(movements)
    for ap in _APPROACHES:
        header_row_1.append(ap)
    header_row_1.append("Total")

    total_cols = 1 + num_mv * len(_APPROACHES) + len(_APPROACHES) + 1

    # ---- Build data rows ----
    data_rows: list[list[str]] = []
    interval_row_totals: Dict[str, int] = {}
    hourly_accum: Dict[str, list[int]] = defaultdict(list)

    for interval in sorted_intervals:
        row: list[str] = [interval]
        row_vehicle_total = 0
        row_heavy_total = 0

        for ap in _APPROACHES:
            for mv in movements:
                total, heavy = lookup.get((interval, ap, mv), (0, 0))
                cell = str(total)
                if heavy > 0:
                    cell += f" ({heavy})"
                row.append(cell)
                row_vehicle_total += total
                row_heavy_total += heavy

        for ap in _APPROACHES:
            row.append(str(ped_lookup.get((interval, ap), 0)))

        row.append(str(row_vehicle_total))
        data_rows.append(row)
        interval_row_totals[interval] = row_vehicle_total

    # ---- Totals row ----
    totals_row: list[str] = ["TOTAL"]
    grand_vehicle = 0
    for ap in _APPROACHES:
        for mv in movements:
            col_total = sum(
                lookup.get((iv, ap, mv), (0, 0))[0] for iv in sorted_intervals
            )
            col_heavy = sum(
                lookup.get((iv, ap, mv), (0, 0))[1] for iv in sorted_intervals
            )
            cell = str(col_total)
            if col_heavy > 0:
                cell += f" ({col_heavy})"
            totals_row.append(cell)
            grand_vehicle += col_total

    for ap in _APPROACHES:
        ped_total = sum(ped_lookup.get((iv, ap), 0) for iv in sorted_intervals)
        totals_row.append(str(ped_total))

    totals_row.append(str(grand_vehicle))
    data_rows.append(totals_row)

    # ---- Assemble full table data ----
    table_data = [header_row_0, header_row_1] + data_rows

    # ---- Column widths ----
    time_col_w = 0.52 * inch
    mv_col_w = 0.56 * inch
    ped_col_w = 0.42 * inch
    total_col_w = 0.55 * inch
    col_widths = (
        [time_col_w]
        + [mv_col_w] * (num_mv * len(_APPROACHES))
        + [ped_col_w] * len(_APPROACHES)
        + [total_col_w]
    )

    # ---- Cell style for compact font ----
    cell_style = rl["ParagraphStyle"](
        "CellStyle", fontSize=7, leading=8, alignment=rl["TA_CENTER"],
    )

    # ---- Build Table ----
    tmc_table = Table(table_data, colWidths=col_widths, repeatRows=2)

    style_cmds: list = [
        # ---- Header rows ----
        ("BACKGROUND", (0, 0), (-1, 0), _HEADER_BG),
        ("BACKGROUND", (0, 1), (-1, 1), _SUBHEADER_BG),
        ("TEXTCOLOR", (0, 0), (-1, 1), _WHITE),
        ("FONTNAME", (0, 0), (-1, 1), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, 1), 7),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        # ---- Body ----
        ("FONTNAME", (0, 2), (-1, -1), "Helvetica"),
        ("FONTSIZE", (0, 2), (-1, -1), 7),
        # ---- Time column left-aligned ----
        ("ALIGN", (0, 2), (0, -1), "LEFT"),
        ("FONTNAME", (0, 2), (0, -1), "Helvetica-Bold"),
        # ---- Grid ----
        ("GRID", (0, 0), (-1, -1), 0.4, _GRID_COLOR),
        ("LINEBELOW", (0, 1), (-1, 1), 1.0, _HEADER_BG),
        # ---- Totals row ----
        ("BACKGROUND", (0, -1), (-1, -1), colors_mod.HexColor("#D5DBDB")),
        ("FONTNAME", (0, -1), (-1, -1), "Helvetica-Bold"),
        # ---- Padding ----
        ("TOPPADDING", (0, 0), (-1, -1), 2),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
        ("LEFTPADDING", (0, 0), (-1, -1), 2),
        ("RIGHTPADDING", (0, 0), (-1, -1), 2),
    ]

    # ---- Span header_row_0 group labels ----
    col = 1
    for i, ap in enumerate(_APPROACHES):
        style_cmds.append(("SPAN", (col, 0), (col + num_mv - 1, 0)))
        col += num_mv
    # Peds group
    style_cmds.append(("SPAN", (col, 0), (col + len(_APPROACHES) - 1, 0)))
    col += len(_APPROACHES)
    # Int Total
    style_cmds.append(("SPAN", (col, 0), (col, 1)))

    # ---- Approach column separators (thicker vertical lines) ----
    sep_col = 1
    for _ in _APPROACHES:
        style_cmds.append(
            ("LINEAFTER", (sep_col + num_mv - 1, 0),
             (sep_col + num_mv - 1, -1), 1.0, _HEADER_BG)
        )
        sep_col += num_mv

    # ---- Highlight peak-hour rows ----
    for row_idx, interval in enumerate(sorted_intervals):
        if interval in peak_intervals:
            table_row = row_idx + 2  # +2 for header rows
            style_cmds.append(
                ("BACKGROUND", (0, table_row), (-1, table_row), _PEAK_ROW_BG)
            )

    tmc_table.setStyle(TableStyle(style_cmds))
    elements.append(tmc_table)

    # ---- Legend ----
    legend_style = rl["ParagraphStyle"](
        "Legend", fontSize=7, leading=9, textColor=colors_mod.HexColor("#555555"),
    )
    elements.append(Spacer(1, 0.12 * inch))
    elements.append(Paragraph(
        "Heavy vehicle counts shown in parentheses (HV).  "
        "Highlighted rows indicate peak-hour intervals.",
        legend_style,
    ))

    return elements


def _parse_peak_intervals(peak_str: str) -> set[str]:
    """Parse a peak-hour string like ``"07:45-08:45"`` into the set of
    constituent 15-min labels: ``{"07:45","08:00","08:15","08:30"}``."""
    if not peak_str or peak_str == "N/A":
        return set()
    try:
        start_s, end_s = peak_str.split("-")
        sh, sm = int(start_s.split(":")[0]), int(start_s.split(":")[1])
        eh, em = int(end_s.split(":")[0]), int(end_s.split(":")[1])
        from datetime import datetime, timedelta
        dt = datetime(2000, 1, 1, sh, sm)
        end_dt = datetime(2000, 1, 1, eh, em)
        labels: set[str] = set()
        while dt < end_dt:
            labels.add(dt.strftime("%H:%M"))
            dt += timedelta(minutes=15)
        return labels
    except Exception:
        return set()


# ---------------------------------------------------------------------------
# Speed summary page
# ---------------------------------------------------------------------------

def _build_speed_page(speed_data: List[Dict[str, Any]], rl: dict) -> list:
    """Build a speed summary page with statistics and a histogram."""
    inch = rl["inch"]
    Paragraph = rl["Paragraph"]
    Spacer = rl["Spacer"]
    Table = rl["Table"]
    TableStyle = rl["TableStyle"]
    colors_mod = rl["colors"]
    styles = rl["getSampleStyleSheet"]()
    Drawing = rl["Drawing"]
    VerticalBarChart = rl["VerticalBarChart"]
    Rect = rl["Rect"]
    String = rl["String"]

    elements: list = []

    heading_style = rl["ParagraphStyle"](
        "SpeedHeading",
        parent=styles["Heading2"],
        fontSize=13,
        textColor=_HEADER_BG,
        spaceAfter=8,
    )
    elements.append(Paragraph("Speed Summary", heading_style))
    elements.append(Spacer(1, 0.1 * inch))

    # ---- Compute statistics ----
    speeds = [d["Speed_KMH"] for d in speed_data if d.get("Speed_KMH", 0) > 0]

    if not speeds:
        elements.append(Paragraph("No speed data available.", styles["Normal"]))
        return elements

    speeds_sorted = sorted(speeds)
    n = len(speeds_sorted)
    mean_speed = statistics.mean(speeds_sorted)
    p85_idx = int(math.ceil(0.85 * n)) - 1
    p85_speed = speeds_sorted[min(p85_idx, n - 1)]
    median_speed = statistics.median(speeds_sorted)

    # Pace: find the 10 km/h range containing the most observations
    pace_lower, pace_count = _compute_pace(speeds_sorted, 10.0)
    pace_str = f"{pace_lower:.0f}–{pace_lower + 10:.0f}"

    # ---- Statistics table ----
    stat_data = [
        ["Metric", "Value"],
        ["Sample Size", f"{n:,}"],
        ["Mean Speed", f"{mean_speed:.1f} km/h"],
        ["Median Speed", f"{median_speed:.1f} km/h"],
        ["85th Percentile Speed", f"{p85_speed:.1f} km/h"],
        ["Pace (10 km/h range)", f"{pace_str} km/h"],
        ["Pace Count", f"{pace_count:,} ({pace_count / n * 100:.1f}%)"],
    ]
    stat_table = Table(stat_data, colWidths=[2.5 * inch, 2.5 * inch])
    stat_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), _HEADER_BG),
        ("TEXTCOLOR", (0, 0), (-1, 0), _WHITE),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 10),
        ("FONTNAME", (0, 1), (0, -1), "Helvetica-Bold"),
        ("ALIGN", (1, 0), (1, -1), "CENTER"),
        ("GRID", (0, 0), (-1, -1), 0.5, _GRID_COLOR),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    elements.append(stat_table)
    elements.append(Spacer(1, 0.35 * inch))

    # ---- Histogram (speed distribution) ----
    elements.append(Paragraph("Speed Distribution", heading_style))
    elements.append(Spacer(1, 0.08 * inch))

    bin_width = 5.0
    min_bin = int(min(speeds_sorted) // bin_width) * int(bin_width)
    max_bin = int(max(speeds_sorted) // bin_width + 1) * int(bin_width) + int(bin_width)
    bin_edges = []
    edge = float(min_bin)
    while edge <= max_bin:
        bin_edges.append(edge)
        edge += bin_width

    hist_counts: list[int] = []
    bin_labels: list[str] = []
    for i in range(len(bin_edges) - 1):
        lo = bin_edges[i]
        hi = bin_edges[i + 1]
        count = sum(1 for s in speeds_sorted if lo <= s < hi)
        hist_counts.append(count)
        bin_labels.append(f"{int(lo)}")

    chart_width = min(8.0 * inch, max(4.0 * inch, len(hist_counts) * 0.45 * inch))
    chart_height = 2.5 * inch
    drawing = Drawing(chart_width + 1.0 * inch, chart_height + 0.8 * inch)

    bc = VerticalBarChart()
    bc.x = 0.6 * inch
    bc.y = 0.5 * inch
    bc.width = chart_width - 0.2 * inch
    bc.height = chart_height - 0.3 * inch
    bc.data = [hist_counts]
    bc.categoryAxis.categoryNames = bin_labels
    bc.categoryAxis.labels.fontSize = 6
    bc.categoryAxis.labels.angle = 0
    bc.valueAxis.valueMin = 0
    bc.valueAxis.labels.fontSize = 7
    bc.bars[0].fillColor = colors_mod.HexColor("#2980B9")
    bc.bars[0].strokeColor = colors_mod.HexColor("#1A5276")
    bc.bars[0].strokeWidth = 0.3
    bc.barWidth = max(4, min(18, int(chart_width / max(len(hist_counts), 1) * 0.6)))

    drawing.add(bc)

    # Axis labels
    drawing.add(String(
        bc.x + bc.width / 2, 0.12 * inch,
        "Speed (km/h)", fontSize=8, textAnchor="middle",
    ))
    drawing.add(String(
        0.15 * inch, bc.y + bc.height / 2,
        "Count", fontSize=8, textAnchor="middle",
    ))

    elements.append(drawing)

    return elements


def _compute_pace(sorted_speeds: List[float], pace_width: float) -> Tuple[float, int]:
    """Find the *pace_width* km/h range that contains the most observations.

    Parameters:
        sorted_speeds: Ascending-sorted list of speeds.
        pace_width:    Width of the pace window (default 10 km/h).

    Returns:
        ``(lower_bound, count)`` of the best pace range.
    """
    if not sorted_speeds:
        return (0.0, 0)

    best_lower = sorted_speeds[0]
    best_count = 0

    # Slide in 1 km/h increments
    min_s = int(math.floor(sorted_speeds[0]))
    max_s = int(math.ceil(sorted_speeds[-1]))

    for lower in range(min_s, max_s + 1):
        upper = lower + pace_width
        count = sum(1 for s in sorted_speeds if lower <= s < upper)
        if count > best_count:
            best_count = count
            best_lower = float(lower)

    return (best_lower, best_count)


# ---------------------------------------------------------------------------
# Page footers
# ---------------------------------------------------------------------------

def _footer_portrait(canvas, doc):
    """Draw a discreet footer on portrait pages."""
    canvas.saveState()
    canvas.setFont("Helvetica", 7)
    canvas.setFillColor("#888888")
    canvas.drawCentredString(
        doc.pagesize[0] / 2, 0.35 * 72,
        f"TrafficIQ — Generated {datetime.now().strftime('%Y-%m-%d %H:%M')}    |    "
        f"Page {canvas.getPageNumber()}",
    )
    canvas.restoreState()


def _footer_landscape(canvas, doc):
    """Draw a discreet footer on landscape pages."""
    canvas.saveState()
    canvas.setFont("Helvetica", 7)
    canvas.setFillColor("#888888")
    canvas.drawCentredString(
        doc.pagesize[0] / 2, 0.3 * 72,
        f"TrafficIQ — Generated {datetime.now().strftime('%Y-%m-%d %H:%M')}    |    "
        f"Page {canvas.getPageNumber()}",
    )
    canvas.restoreState()


# ===================================================================
# 2. UTDF CSV (Synchro-compatible Volume.csv)
# ===================================================================

def generate_utdf_csv(
    result: TrafficStudyResult,
    output_path: str,
    intersection_id: int = 1,
) -> str:
    """Generate a Synchro-compatible UTDF ``Volume.csv``.

    The output follows the UTDF 15-Minute Counts format::

        Volume Data
        15 Minute Counts
        INTID,NBL,NBT,NBR,SBL,SBT,SBR,EBL,EBT,EBR,WBL,WBT,WBR,\\
            Ped_NB,Ped_SB,Ped_EB,Ped_WB,Date,Day,Time
        1,5,32,8,...,03/12/2026,Thu,07:00
        ...

    Parameters:
        result:          A :class:`~interval_binning.TrafficStudyResult`.
        output_path:     Filesystem path for the generated CSV.
        intersection_id: INTID value to write (defaults to ``1``).

    Returns:
        The *output_path* string.
    """
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    # ---- Pre-aggregate ----
    sorted_intervals = sorted(
        {b.interval_start for b in result.bins},
        key=lambda s: (int(s.split(":")[0]), int(s.split(":")[1])),
    )

    # (interval, approach, movement) → vehicle total
    lookup: Dict[Tuple[str, str, str], int] = {}
    ped_lookup: Dict[Tuple[str, str], int] = defaultdict(int)

    for b in result.bins:
        lookup[(b.interval_start, b.approach, b.movement)] = b.total
        # Pedestrian counts may be duplicated across movements; take the max
        ped_lookup[(b.interval_start, b.approach)] = max(
            ped_lookup[(b.interval_start, b.approach)], b.pedestrian_count,
        )

    # ---- Date and day parsing ----
    try:
        study_dt = datetime.strptime(result.study_date, "%Y-%m-%d")
    except ValueError:
        study_dt = datetime.now()
    date_str = study_dt.strftime("%m/%d/%Y")
    day_str = study_dt.strftime("%a")  # Mon, Tue, …

    # ---- Movement column order ----
    movements = ["L", "T", "R"]
    col_headers = []
    for ap in _APPROACHES:
        for mv in movements:
            col_headers.append(f"{ap}{mv}")
    for ap in _APPROACHES:
        col_headers.append(f"Ped_{ap}")

    header = ["INTID"] + col_headers + ["Date", "Day", "Time"]

    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Volume Data"])
        writer.writerow(["15 Minute Counts"])
        writer.writerow(header)

        for interval in sorted_intervals:
            row: list = [intersection_id]
            for ap in _APPROACHES:
                for mv in movements:
                    row.append(lookup.get((interval, ap, mv), 0))
            for ap in _APPROACHES:
                row.append(ped_lookup.get((interval, ap), 0))
            row.extend([date_str, day_str, interval])
            writer.writerow(row)

    return output_path


# ===================================================================
# 3. Speed Summary CSV
# ===================================================================

def generate_speed_summary_csv(
    speeds: List[Dict[str, Any]],
    output_path: str,
) -> str:
    """Generate a per-vehicle speed log CSV with a statistical summary footer.

    Expected keys in each dict:
        ``Track_ID``, ``Class``, ``Speed_KMH``, ``Timestamp_S``, ``Approach``

    Output format::

        Track_ID,Class,Speed_KMH,Timestamp_S,Approach
        1,Passenger Vehicle,52.3,12.45,NB
        ...

        Summary
        85th Percentile Speed,52.8
        Mean Speed,45.2
        Pace (10 km/h range),40-50
        Sample Size,847

    Parameters:
        speeds:      List of per-vehicle speed dictionaries.
        output_path: Filesystem path for the generated CSV.

    Returns:
        The *output_path* string.
    """
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    header = ["Track_ID", "Class", "Speed_KMH", "Timestamp_S", "Approach"]

    speed_values = [
        d["Speed_KMH"] for d in speeds if d.get("Speed_KMH", 0) > 0
    ]
    speed_values_sorted = sorted(speed_values)

    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)

        for rec in speeds:
            writer.writerow([
                rec.get("Track_ID", ""),
                rec.get("Class", ""),
                rec.get("Speed_KMH", ""),
                rec.get("Timestamp_S", ""),
                rec.get("Approach", ""),
            ])

        # Blank line separator
        writer.writerow([])

        # Summary section
        writer.writerow(["Summary"])

        n = len(speed_values_sorted)
        if n > 0:
            mean_speed = statistics.mean(speed_values_sorted)
            p85_idx = int(math.ceil(0.85 * n)) - 1
            p85_speed = speed_values_sorted[min(p85_idx, n - 1)]
            pace_lower, _pace_count = _compute_pace(speed_values_sorted, 10.0)
            pace_str = f"{pace_lower:.0f}-{pace_lower + 10:.0f}"
        else:
            mean_speed = 0.0
            p85_speed = 0.0
            pace_str = "N/A"

        writer.writerow(["85th Percentile Speed", f"{p85_speed:.1f}"])
        writer.writerow(["Mean Speed", f"{mean_speed:.1f}"])
        writer.writerow(["Pace (10 km/h range)", pace_str])
        writer.writerow(["Sample Size", str(n)])

    return output_path
