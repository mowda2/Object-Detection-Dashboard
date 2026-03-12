"""
Interval Binning Module for TrafficIQ

Aggregates :class:`~turn_movements.CompletedMovement` and
:class:`~counting.CrossingEvent` objects into clock-aligned 15-minute bins
and computes standard traffic engineering metrics (peak hours, PHF, heavy
vehicle percentage, etc.).

Bin boundaries are always aligned to wall-clock quarter-hours
(``XX:00``, ``XX:15``, ``XX:30``, ``XX:45``).  A ``study_start_time``
parameter maps *video second 0* to a real-world time-of-day so that every
event can be assigned to the correct clock-time bin.

Classes:
    IntervalBin           – one 15-min interval's data for one approach
                            and movement type
    TrafficStudyResult    – complete study output with computed metrics
    IntervalAggregator    – stateful accumulator; feed it movements /
                            crossings, then call ``compute_results()``

Typical usage::

    from turn_movements import TurnMovementClassifier, CompletedMovement
    from interval_binning import IntervalAggregator

    agg = IntervalAggregator(study_start_time="07:00:00")
    for m in completed_movements:
        agg.add_movement(m)
    for ped in ped_crossings:
        agg.add_pedestrian(ped)
    for bike in bike_crossings:
        agg.add_bicycle(bike)

    result = agg.compute_results("Main St & 1st Ave", "2026-03-12")
    print(result.am_peak_hour, result.am_peak_phf)
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

from .counting import CrossingEvent
from .turn_movements import CompletedMovement, TurnMovement


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Canonical ordering of motorised vehicle classes (excludes Pedestrian and
#: Bicycle which are tracked separately).
VEHICLE_CLASSES: List[str] = [
    "Passenger Vehicle",
    "Light Truck",
    "Heavy Truck",
    "Bus",
    "Motorcycle",
]

#: Classes that count towards the "heavy vehicle" total.
_HEAVY_CLASSES: frozenset[str] = frozenset({"Heavy Truck", "Bus"})

#: Canonical approach labels.
_APPROACHES: List[str] = ["NB", "SB", "EB", "WB"]

#: Canonical movement labels.
_MOVEMENTS: List[str] = ["L", "T", "R", "U"]

#: Duration of one bin in minutes.
_BIN_MINUTES: int = 15

#: Number of bins that make up one hour (used for peak-hour sliding window).
_BINS_PER_HOUR: int = 4


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class IntervalBin:
    """One 15-minute interval's data for a single approach and movement type.

    Attributes:
        interval_start:     Clock-time label, e.g. ``"07:15"``.
        approach:           ``"NB"``, ``"SB"``, ``"EB"``, or ``"WB"``.
        movement:           ``"L"``, ``"T"``, ``"R"``, or ``"U"``.
        vehicle_counts:     Per-class counts for motorised vehicles.
        total:              Sum of all motorised vehicle counts.
        heavy_vehicle_count: Sum of ``Heavy Truck`` + ``Bus``.
        pedestrian_count:   Pedestrians binned to this interval/approach.
        bicycle_count:      Bicycles binned to this interval/approach.
    """

    interval_start: str
    approach: str
    movement: str
    vehicle_counts: Dict[str, int]
    total: int
    heavy_vehicle_count: int
    pedestrian_count: int
    bicycle_count: int


@dataclass
class TrafficStudyResult:
    """Complete output of a traffic study analysis.

    Attributes:
        intersection_name:  Human-readable intersection label.
        study_date:         Date string, e.g. ``"2026-03-12"``.
        study_start:        First bin label, e.g. ``"07:00"``.
        study_end:          Label of the bin *after* the last one, e.g.
                            ``"19:00"``.
        bins:               All :class:`IntervalBin` objects in chronological
                            order.
        am_peak_hour:       AM peak hour range, e.g. ``"07:45-08:45"``.
        am_peak_volume:     Total volume during the AM peak hour.
        am_peak_phf:        Peak Hour Factor for the AM peak.
        pm_peak_hour:       PM peak hour range, e.g. ``"16:30-17:30"``.
        pm_peak_volume:     Total volume during the PM peak hour.
        pm_peak_phf:        Peak Hour Factor for the PM peak.
        total_volume:       Sum of all motorised vehicle counts across the
                            study period.
        heavy_vehicle_pct:  Heavy-vehicle percentage (0–100).
    """

    intersection_name: str
    study_date: str
    study_start: str
    study_end: str
    bins: List[IntervalBin]

    am_peak_hour: str
    am_peak_volume: int
    am_peak_phf: float
    pm_peak_hour: str
    pm_peak_volume: int
    pm_peak_phf: float
    total_volume: int
    heavy_vehicle_pct: float


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _parse_time(t: str) -> datetime:
    """Parse ``"HH:MM"`` or ``"HH:MM:SS"`` into a :class:`datetime` (date
    portion is arbitrary — only the time component matters)."""
    fmt = "%H:%M:%S" if t.count(":") == 2 else "%H:%M"
    return datetime.strptime(t, fmt)


def _floor_to_quarter(dt: datetime) -> datetime:
    """Round *dt* down to the nearest quarter-hour boundary."""
    minute = (dt.minute // _BIN_MINUTES) * _BIN_MINUTES
    return dt.replace(minute=minute, second=0, microsecond=0)


def _time_label(dt: datetime) -> str:
    """Format a datetime as ``"HH:MM"``."""
    return dt.strftime("%H:%M")


def _seconds_to_datetime(base: datetime, seconds: float) -> datetime:
    """Add *seconds* to a base datetime."""
    return base + timedelta(seconds=seconds)


# ---------------------------------------------------------------------------
# Aggregator
# ---------------------------------------------------------------------------

class IntervalAggregator:
    """Accumulates movements and crossings into clock-aligned 15-minute bins.

    Parameters:
        study_start_time: Real-world time-of-day corresponding to *second 0*
                          of the video.  Accepts ``"HH:MM"`` or
                          ``"HH:MM:SS"``.  Defaults to ``"00:00:00"``.
    """

    VEHICLE_CLASSES = VEHICLE_CLASSES  # expose at class level for external use

    def __init__(self, study_start_time: str = "00:00:00") -> None:
        self._base_dt: datetime = _parse_time(study_start_time)

        # (bin_label, approach, movement) → {vehicle_class: count}
        self._vehicle_bins: Dict[
            Tuple[str, str, str], Dict[str, int]
        ] = defaultdict(lambda: {vc: 0 for vc in VEHICLE_CLASSES})

        # (bin_label, approach) → pedestrian count
        self._ped_bins: Dict[Tuple[str, str], int] = defaultdict(int)

        # (bin_label, approach) → bicycle count
        self._bike_bins: Dict[Tuple[str, str], int] = defaultdict(int)

        # Track the actual range of bin labels seen.
        self._seen_labels: set[str] = set()

    # ------------------------------------------------------------------
    # Time conversion
    # ------------------------------------------------------------------

    def _video_seconds_to_bin_label(self, seconds: float) -> str:
        """Convert a video-relative timestamp to a clock-aligned bin label."""
        real_dt = _seconds_to_datetime(self._base_dt, seconds)
        bin_dt = _floor_to_quarter(real_dt)
        label = _time_label(bin_dt)
        self._seen_labels.add(label)
        return label

    # ------------------------------------------------------------------
    # Ingestion
    # ------------------------------------------------------------------

    def add_movement(self, movement: CompletedMovement) -> None:
        """Add a completed vehicle movement to the appropriate bin.

        The movement is binned by its *entry* time (when the vehicle entered
        the intersection).

        Parameters:
            movement: A :class:`~turn_movements.CompletedMovement`.
        """
        label = self._video_seconds_to_bin_label(movement.entry_time_s)
        approach = movement.entry_approach
        mv = movement.movement.value if isinstance(
            movement.movement, TurnMovement
        ) else str(movement.movement)

        key = (label, approach, mv)
        class_counts = self._vehicle_bins[key]

        vc = movement.vehicle_class
        if vc in class_counts:
            class_counts[vc] += 1
        else:
            # Unknown class — accumulate under "Other" or add ad-hoc.
            class_counts.setdefault(vc, 0)
            class_counts[vc] += 1

    def add_pedestrian(self, crossing: CrossingEvent) -> None:
        """Bin a pedestrian crossing event.

        Pedestrians are counted per approach (the line/crosswalk they crossed)
        and 15-minute interval.

        Parameters:
            crossing: A :class:`~counting.CrossingEvent` whose
                      ``object_class`` is ``"Pedestrian"``.
        """
        label = self._video_seconds_to_bin_label(crossing.timestamp_s)
        self._ped_bins[(label, crossing.direction)] += 1

    def add_bicycle(self, crossing: CrossingEvent) -> None:
        """Bin a bicycle crossing event.

        Parameters:
            crossing: A :class:`~counting.CrossingEvent` whose
                      ``object_class`` is ``"Bicycle"``.
        """
        label = self._video_seconds_to_bin_label(crossing.timestamp_s)
        self._bike_bins[(label, crossing.direction)] += 1

    # ------------------------------------------------------------------
    # Result computation
    # ------------------------------------------------------------------

    def compute_results(
        self,
        intersection_name: str,
        study_date: str,
    ) -> TrafficStudyResult:
        """Build all :class:`IntervalBin` objects and compute peak-hour
        metrics.

        Parameters:
            intersection_name: Human-readable label for the intersection.
            study_date:        Date string (e.g. ``"2026-03-12"``).

        Returns:
            A fully-populated :class:`TrafficStudyResult`.
        """
        # ---- 1. Determine the full range of 15-min labels ----
        sorted_labels = self._sorted_bin_labels()
        if not sorted_labels:
            # No data at all — return an empty result.
            return self._empty_result(intersection_name, study_date)

        study_start = sorted_labels[0]
        last_label_dt = _parse_time(sorted_labels[-1])
        study_end_dt = last_label_dt + timedelta(minutes=_BIN_MINUTES)
        study_end = _time_label(study_end_dt)

        # ---- 2. Build IntervalBin list ----
        bins: List[IntervalBin] = []
        for label in sorted_labels:
            for approach in _APPROACHES:
                for mv in _MOVEMENTS:
                    key = (label, approach, mv)
                    class_counts = dict(self._vehicle_bins.get(
                        key,
                        {vc: 0 for vc in VEHICLE_CLASSES},
                    ))
                    total = sum(class_counts.values())
                    heavy = sum(
                        class_counts.get(hc, 0) for hc in _HEAVY_CLASSES
                    )
                    ped = self._ped_bins.get((label, approach), 0)
                    bike = self._bike_bins.get((label, approach), 0)

                    bins.append(IntervalBin(
                        interval_start=label,
                        approach=approach,
                        movement=mv,
                        vehicle_counts=class_counts,
                        total=total,
                        heavy_vehicle_count=heavy,
                        pedestrian_count=ped,
                        bicycle_count=bike,
                    ))

        # ---- 3. Per-interval aggregate volumes (all approaches, all moves) -
        interval_totals: Dict[str, int] = defaultdict(int)
        interval_heavy: Dict[str, int] = defaultdict(int)
        for b in bins:
            interval_totals[b.interval_start] += b.total
            interval_heavy[b.interval_start] += b.heavy_vehicle_count

        total_volume = sum(interval_totals.values())
        total_heavy = sum(interval_heavy.values())
        heavy_vehicle_pct = (
            (total_heavy / total_volume * 100.0) if total_volume > 0 else 0.0
        )

        # ---- 4. Peak-hour analysis (sliding 4-bin window) ----
        am_peak_hour, am_peak_volume, am_peak_phf = self._find_peak(
            sorted_labels, interval_totals, "06:00", "10:00",
        )
        pm_peak_hour, pm_peak_volume, pm_peak_phf = self._find_peak(
            sorted_labels, interval_totals, "15:00", "19:00",
        )

        return TrafficStudyResult(
            intersection_name=intersection_name,
            study_date=study_date,
            study_start=study_start,
            study_end=study_end,
            bins=bins,
            am_peak_hour=am_peak_hour,
            am_peak_volume=am_peak_volume,
            am_peak_phf=am_peak_phf,
            pm_peak_hour=pm_peak_hour,
            pm_peak_volume=pm_peak_volume,
            pm_peak_phf=pm_peak_phf,
            total_volume=total_volume,
            heavy_vehicle_pct=round(heavy_vehicle_pct, 2),
        )

    # ------------------------------------------------------------------
    # Peak-hour helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _find_peak(
        sorted_labels: List[str],
        interval_totals: Dict[str, int],
        window_start: str,
        window_end: str,
    ) -> Tuple[str, int, float]:
        """Slide a 1-hour (4-bin) window within *[window_start, window_end)*
        and return the peak hour, its volume, and its PHF.

        Parameters:
            sorted_labels:   All bin labels in chronological order.
            interval_totals: ``{label: total_vehicle_count}``.
            window_start:    Earliest label to consider (inclusive),
                             e.g. ``"06:00"``.
            window_end:      Latest label to consider (exclusive),
                             e.g. ``"10:00"``.

        Returns:
            ``(peak_hour_range, peak_volume, phf)``  where
            *peak_hour_range* is e.g. ``"07:45-08:45"`` and *phf* is clamped
            to ``[0, 1]``.  If no data falls in the window, returns
            ``("N/A", 0, 0.0)``.
        """
        ws_dt = _parse_time(window_start)
        we_dt = _parse_time(window_end)

        # Filter labels that fall within the search window.
        eligible: List[str] = []
        for lbl in sorted_labels:
            lbl_dt = _parse_time(lbl)
            if ws_dt <= lbl_dt < we_dt:
                eligible.append(lbl)

        if len(eligible) < _BINS_PER_HOUR:
            # Not enough data for a full 1-hour window.
            if not eligible:
                return ("N/A", 0, 0.0)
            # Use whatever is available.
            vol = sum(interval_totals.get(l, 0) for l in eligible)
            max_bin = max(interval_totals.get(l, 0) for l in eligible)
            n = len(eligible)
            phf = (vol / (n * max_bin)) if max_bin > 0 else 0.0
            first = eligible[0]
            last_dt = _parse_time(eligible[-1]) + timedelta(minutes=_BIN_MINUTES)
            return (f"{first}-{_time_label(last_dt)}", vol, round(phf, 3))

        best_vol = -1
        best_start_idx = 0

        for i in range(len(eligible) - _BINS_PER_HOUR + 1):
            window_labels = eligible[i : i + _BINS_PER_HOUR]
            vol = sum(interval_totals.get(l, 0) for l in window_labels)
            if vol > best_vol:
                best_vol = vol
                best_start_idx = i

        peak_labels = eligible[best_start_idx : best_start_idx + _BINS_PER_HOUR]
        peak_volumes = [interval_totals.get(l, 0) for l in peak_labels]
        peak_volume = sum(peak_volumes)
        max_single = max(peak_volumes)

        # PHF = V / (4 × V_max_15)
        phf = (peak_volume / (_BINS_PER_HOUR * max_single)) if max_single > 0 else 0.0

        peak_start = peak_labels[0]
        peak_end_dt = _parse_time(peak_labels[-1]) + timedelta(minutes=_BIN_MINUTES)
        peak_range = f"{peak_start}-{_time_label(peak_end_dt)}"

        return (peak_range, peak_volume, round(phf, 3))

    # ------------------------------------------------------------------
    # Convenience / query helpers
    # ------------------------------------------------------------------

    def get_approach_summary(self) -> Dict[str, Dict[str, int]]:
        """Return total counts grouped by approach and movement across all
        bins.

        Returns:
            ``{approach: {movement: total_count}}``.

        Example::

            {
                "NB": {"L": 48, "T": 210, "R": 62, "U": 2},
                "SB": {"L": 31, "T": 195, "R": 55, "U": 0},
                ...
            }
        """
        summary: Dict[str, Dict[str, int]] = {
            ap: {mv: 0 for mv in _MOVEMENTS} for ap in _APPROACHES
        }
        for (label, approach, mv), class_counts in self._vehicle_bins.items():
            total = sum(class_counts.values())
            summary[approach][mv] += total
        return summary

    def get_interval_totals(self) -> Dict[str, int]:
        """Return the aggregate vehicle count for each 15-min interval
        (all approaches and movements combined).

        Returns:
            ``{bin_label: total_count}`` in chronological order.
        """
        totals: Dict[str, int] = defaultdict(int)
        for (label, _approach, _mv), class_counts in self._vehicle_bins.items():
            totals[label] += sum(class_counts.values())
        # Return sorted by time.
        return dict(sorted(totals.items(), key=lambda kv: _parse_time(kv[0])))

    def get_pedestrian_totals(self) -> Dict[str, Dict[str, int]]:
        """Return pedestrian counts by interval and approach.

        Returns:
            ``{bin_label: {approach: count}}``.
        """
        result: Dict[str, Dict[str, int]] = defaultdict(
            lambda: {ap: 0 for ap in _APPROACHES}
        )
        for (label, approach), count in self._ped_bins.items():
            result[label][approach] += count
        return dict(sorted(result.items(), key=lambda kv: _parse_time(kv[0])))

    def get_bicycle_totals(self) -> Dict[str, Dict[str, int]]:
        """Return bicycle counts by interval and approach.

        Returns:
            ``{bin_label: {approach: count}}``.
        """
        result: Dict[str, Dict[str, int]] = defaultdict(
            lambda: {ap: 0 for ap in _APPROACHES}
        )
        for (label, approach), count in self._bike_bins.items():
            result[label][approach] += count
        return dict(sorted(result.items(), key=lambda kv: _parse_time(kv[0])))

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _sorted_bin_labels(self) -> List[str]:
        """Return all observed bin labels sorted chronologically."""
        return sorted(self._seen_labels, key=lambda l: _parse_time(l))

    def _empty_result(
        self, intersection_name: str, study_date: str
    ) -> TrafficStudyResult:
        """Construct a :class:`TrafficStudyResult` with zero data."""
        return TrafficStudyResult(
            intersection_name=intersection_name,
            study_date=study_date,
            study_start="N/A",
            study_end="N/A",
            bins=[],
            am_peak_hour="N/A",
            am_peak_volume=0,
            am_peak_phf=0.0,
            pm_peak_hour="N/A",
            pm_peak_volume=0,
            pm_peak_phf=0.0,
            total_volume=0,
            heavy_vehicle_pct=0.0,
        )
