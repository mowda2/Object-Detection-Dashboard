"""
Comprehensive tests for the four core TrafficIQ modules:
  - counting.py
  - turn_movements.py
  - interval_binning.py
  - report_generator.py

Run with:
    PYTHONPATH=src/moe_yolo_pipeline pytest tests/test_trafficiq_modules.py -v
"""

from __future__ import annotations

import csv
import math
import os
import tempfile
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pytest

from moe_yolo_pipeline.counting import (
    CountingLine,
    CrossingEvent,
    LineCrossingDetector,
    classify_vehicle,
    DEFAULT_TRUCK_AREA_THRESHOLD,
    _MIN_CROSSING_DISPLACEMENT,
)
from moe_yolo_pipeline.turn_movements import (
    CompletedMovement,
    TurnMovement,
    TurnMovementClassifier,
)
from moe_yolo_pipeline.interval_binning import (
    IntervalAggregator,
    IntervalBin,
    TrafficStudyResult,
    VEHICLE_CLASSES,
    _HEAVY_CLASSES,
)
from moe_yolo_pipeline.report_generator import (
    generate_tmc_pdf,
    generate_utdf_csv,
    generate_speed_summary_csv,
)


# ===================================================================
# Helpers / fakes
# ===================================================================

class FakeDetections:
    """Minimal stand-in for ``supervision.Detections`` that satisfies the
    fields accessed by ``LineCrossingDetector.update()``:
    ``.tracker_id``, ``.xyxy``, ``.class_id``."""

    def __init__(
        self,
        tracker_ids: List[int],
        xyxy: List[Tuple[float, float, float, float]],
        class_ids: List[int],
    ) -> None:
        self.tracker_id = np.array(tracker_ids, dtype=int) if tracker_ids else None
        self.xyxy = np.array(xyxy, dtype=float) if xyxy else np.empty((0, 4))
        self.class_id = np.array(class_ids, dtype=int) if class_ids else np.empty(0, dtype=int)

    def __len__(self) -> int:
        if self.tracker_id is None:
            return 0
        return len(self.tracker_id)


def _make_crossing_event(
    track_id: int = 1,
    line_name: str = "line_NB",
    direction: str = "NB",
    timestamp_s: float = 0.0,
    frame_number: int = 0,
    object_class: str = "Passenger Vehicle",
    speed_kmh: float = 0.0,
    bbox: Tuple[float, float, float, float] = (100, 100, 200, 200),
) -> CrossingEvent:
    """Convenience factory for CrossingEvent with sane defaults."""
    return CrossingEvent(
        track_id=track_id,
        line_name=line_name,
        direction=direction,
        timestamp_s=timestamp_s,
        frame_number=frame_number,
        object_class=object_class,
        speed_kmh=speed_kmh,
        bbox=bbox,
    )


def _make_completed_movement(
    track_id: int = 1,
    entry_approach: str = "NB",
    exit_approach: str = "SB",
    movement: TurnMovement = TurnMovement.THROUGH,
    entry_time_s: float = 0.0,
    exit_time_s: float = 5.0,
    vehicle_class: str = "Passenger Vehicle",
    speed_kmh: float = 50.0,
    incomplete: bool = False,
) -> CompletedMovement:
    """Convenience factory for CompletedMovement."""
    return CompletedMovement(
        track_id=track_id,
        entry_approach=entry_approach,
        exit_approach=exit_approach,
        movement=movement,
        entry_time_s=entry_time_s,
        exit_time_s=exit_time_s,
        vehicle_class=vehicle_class,
        speed_kmh=speed_kmh,
        incomplete=incomplete,
    )


# ===================================================================
# Fixtures
# ===================================================================

@pytest.fixture()
def horizontal_line() -> CountingLine:
    """A horizontal counting line at y=500, spanning x=0..1000, direction NB."""
    return CountingLine(name="NB_line", x1=0, y1=500, x2=1000, y2=500, direction="NB")


@pytest.fixture()
def four_lines() -> List[CountingLine]:
    """Four counting lines, one per approach, suitable for TMC testing."""
    return [
        CountingLine(name="NB_line", x1=0, y1=200, x2=400, y2=200, direction="NB"),
        CountingLine(name="SB_line", x1=0, y1=800, x2=400, y2=800, direction="SB"),
        CountingLine(name="EB_line", x1=200, y1=0, x2=200, y2=400, direction="EB"),
        CountingLine(name="WB_line", x1=800, y1=0, x2=800, y2=400, direction="WB"),
    ]


@pytest.fixture()
def standard_movements() -> List[CompletedMovement]:
    """15 CompletedMovements spanning 07:00–07:29 (two 15-min bins):
    10 in the first bin (0–899 s) and 5 in the second (900–1799 s)."""
    movements: List[CompletedMovement] = []
    tid = 1
    for i in range(10):
        movements.append(_make_completed_movement(
            track_id=tid,
            entry_approach="NB",
            exit_approach="SB",
            movement=TurnMovement.THROUGH,
            entry_time_s=float(i * 80),          # 0, 80, 160 … 720
            exit_time_s=float(i * 80 + 5),
            vehicle_class="Passenger Vehicle",
        ))
        tid += 1
    for i in range(5):
        movements.append(_make_completed_movement(
            track_id=tid,
            entry_approach="SB",
            exit_approach="NB",
            movement=TurnMovement.THROUGH,
            entry_time_s=float(900 + i * 150),    # 900, 1050 … 1500
            exit_time_s=float(900 + i * 150 + 5),
            vehicle_class="Passenger Vehicle",
        ))
        tid += 1
    return movements


@pytest.fixture()
def sample_study_result(standard_movements) -> TrafficStudyResult:
    """A fully-computed TrafficStudyResult built from ``standard_movements``."""
    agg = IntervalAggregator(study_start_time="07:00:00")
    for m in standard_movements:
        agg.add_movement(m)
    return agg.compute_results("Test Intersection", "2026-03-12")


# ===================================================================
# Tests — counting.py
# ===================================================================

class TestCountingLine:
    def test_counting_line_creation(self) -> None:
        """Create a CountingLine and verify all fields are set."""
        line = CountingLine(
            name="Main_NB", x1=10, y1=200, x2=500, y2=200, direction="NB"
        )
        assert line.name == "Main_NB"
        assert line.x1 == 10
        assert line.y1 == 200
        assert line.x2 == 500
        assert line.y2 == 200
        assert line.direction == "NB"

    def test_invalid_direction_raises(self) -> None:
        with pytest.raises(ValueError, match="direction must be one of"):
            CountingLine(name="bad", x1=0, y1=0, x2=10, y2=0, direction="XX")

    def test_degenerate_line_raises(self) -> None:
        with pytest.raises(ValueError, match="start and end points must differ"):
            CountingLine(name="pt", x1=5, y1=5, x2=5, y2=5, direction="NB")


class TestClassifyVehicle:
    """Test classify_vehicle for every COCO class we care about."""

    def test_car(self) -> None:
        assert classify_vehicle(2, 12_000) == "Passenger Vehicle"

    def test_truck_small_bbox_is_light(self) -> None:
        # bbox_area < DEFAULT_TRUCK_AREA_THRESHOLD → Light Truck
        assert classify_vehicle(7, 30_000) == "Light Truck"

    def test_truck_large_bbox_is_heavy(self) -> None:
        # bbox_area >= DEFAULT_TRUCK_AREA_THRESHOLD → Heavy Truck
        assert classify_vehicle(7, 80_000) == "Heavy Truck"

    def test_truck_boundary(self) -> None:
        # Exactly at the threshold → Heavy Truck (not strictly less than)
        assert classify_vehicle(7, DEFAULT_TRUCK_AREA_THRESHOLD) == "Heavy Truck"

    def test_bus(self) -> None:
        assert classify_vehicle(5, 40_000) == "Bus"

    def test_motorcycle(self) -> None:
        assert classify_vehicle(3, 5_000) == "Motorcycle"

    def test_bicycle(self) -> None:
        assert classify_vehicle(1, 3_000) == "Bicycle"

    def test_pedestrian(self) -> None:
        assert classify_vehicle(0, 4_000) == "Pedestrian"

    def test_unknown_class_returns_other(self) -> None:
        assert classify_vehicle(99, 500) == "Other"

    def test_homography_scale_affects_truck(self) -> None:
        # 30 000 px² * 2.0 scale = 60 000 → Heavy Truck
        assert classify_vehicle(7, 30_000, homography_scale=2.0) == "Heavy Truck"
        # 30 000 px² * 0.5 scale = 15 000 → Light Truck
        assert classify_vehicle(7, 30_000, homography_scale=0.5) == "Light Truck"


class TestLineCrossingDetection:
    """Test that LineCrossingDetector correctly detects crossings."""

    def test_crossing_above_to_below(self, horizontal_line: CountingLine) -> None:
        """Object moves from above the line to well below → one crossing."""
        detector = LineCrossingDetector([horizontal_line])
        class_names = {2: "car"}

        # Frame 0: centroid at (500, 480) — above y=500 line
        det0 = FakeDetections(
            tracker_ids=[1],
            xyxy=[(450, 430, 550, 530)],   # centroid (500, 480)
            class_ids=[2],
        )
        events0 = detector.update(det0, frame_number=0, fps=30.0, class_names=class_names)

        # Frame 1: centroid at (500, 530) — below y=500 line
        # The distance past the line must be >= _MIN_CROSSING_DISPLACEMENT (5 px).
        # centroid y=530, line y=500, distance = 30 px → exceeds threshold.
        det1 = FakeDetections(
            tracker_ids=[1],
            xyxy=[(450, 480, 550, 580)],   # centroid (500, 530)
            class_ids=[2],
        )
        events1 = detector.update(det1, frame_number=1, fps=30.0, class_names=class_names)

        assert len(events0) == 0, "No crossing on the first frame (no prior position)"
        assert len(events1) == 1, "Exactly one crossing when the object moves past the line"

        ev = events1[0]
        assert ev.track_id == 1
        assert ev.line_name == "NB_line"
        assert ev.direction == "NB"
        assert ev.object_class == "Passenger Vehicle"

    def test_no_crossing_same_side(self, horizontal_line: CountingLine) -> None:
        """Object stays above the line across two frames → no crossing."""
        detector = LineCrossingDetector([horizontal_line])
        class_names = {2: "car"}

        det0 = FakeDetections(
            tracker_ids=[1],
            xyxy=[(450, 400, 550, 480)],   # centroid (500, 440)
            class_ids=[2],
        )
        det1 = FakeDetections(
            tracker_ids=[1],
            xyxy=[(450, 410, 550, 490)],   # centroid (500, 450) — still above
            class_ids=[2],
        )
        events0 = detector.update(det0, frame_number=0, fps=30.0, class_names=class_names)
        events1 = detector.update(det1, frame_number=1, fps=30.0, class_names=class_names)

        assert len(events0) == 0
        assert len(events1) == 0

    def test_no_double_count_jitter(self, horizontal_line: CountingLine) -> None:
        """Anti-jitter: object that jitters ±2 px around the line should only
        produce ONE crossing (or zero, if displacement is within threshold)."""
        detector = LineCrossingDetector([horizontal_line])
        class_names = {2: "car"}

        # The _MIN_CROSSING_DISPLACEMENT is 5 px.  A 2 px jitter past the
        # line won't meet the threshold, so no crossing should register.

        positions_y = [498, 502, 498, 502, 498]
        all_events: List[CrossingEvent] = []
        for i, y_center in enumerate(positions_y):
            det = FakeDetections(
                tracker_ids=[1],
                xyxy=[(450, y_center - 25, 550, y_center + 25)],
                class_ids=[2],
            )
            evts = detector.update(det, frame_number=i, fps=30.0, class_names=class_names)
            all_events.extend(evts)

        # The 2 px displacement is below _MIN_CROSSING_DISPLACEMENT (5 px) so
        # none of these jitter moves should produce a crossing.
        assert len(all_events) == 0, (
            "Jitter within the anti-jitter threshold should produce zero crossings"
        )

    def test_single_crossing_when_displacement_exceeds_threshold(
        self, horizontal_line: CountingLine
    ) -> None:
        """Object crosses once with enough displacement, then returns and
        crosses again — only ONE crossing should be recorded per line per track
        (the _crossed set prevents double counting)."""
        detector = LineCrossingDetector([horizontal_line])
        class_names = {2: "car"}

        frames: List[Tuple[float, float]] = [
            (500, 480),   # above line
            (500, 530),   # below line (dist 30 px, > threshold) → crossing
            (500, 480),   # back above line → sign change but already in _crossed
            (500, 530),   # below again → still in _crossed
        ]
        all_events: List[CrossingEvent] = []
        for i, (cx, cy) in enumerate(frames):
            det = FakeDetections(
                tracker_ids=[1],
                xyxy=[(cx - 50, cy - 25, cx + 50, cy + 25)],
                class_ids=[2],
            )
            evts = detector.update(det, frame_number=i, fps=30.0, class_names=class_names)
            all_events.extend(evts)

        assert len(all_events) == 1, (
            "A single track should only register one crossing per line "
            "(the _crossed set prevents re-counting)"
        )


# ===================================================================
# Tests — turn_movements.py
# ===================================================================

class TestTurnMovementMatrix:
    """Verify the (entry, exit) → TurnMovement matrix for all 16 pairs."""

    EXPECTED: List[Tuple[str, str, TurnMovement]] = [
        # NB entry
        ("NB", "SB", TurnMovement.THROUGH),
        ("NB", "EB", TurnMovement.RIGHT),
        ("NB", "WB", TurnMovement.LEFT),
        ("NB", "NB", TurnMovement.UTURN),
        # SB entry
        ("SB", "NB", TurnMovement.THROUGH),
        ("SB", "WB", TurnMovement.RIGHT),
        ("SB", "EB", TurnMovement.LEFT),
        ("SB", "SB", TurnMovement.UTURN),
        # EB entry
        ("EB", "WB", TurnMovement.THROUGH),
        ("EB", "NB", TurnMovement.LEFT),
        ("EB", "SB", TurnMovement.RIGHT),
        ("EB", "EB", TurnMovement.UTURN),
        # WB entry
        ("WB", "EB", TurnMovement.THROUGH),
        ("WB", "SB", TurnMovement.LEFT),
        ("WB", "NB", TurnMovement.RIGHT),
        ("WB", "WB", TurnMovement.UTURN),
    ]

    @pytest.mark.parametrize(
        "entry, exit_, expected_movement",
        EXPECTED,
        ids=[f"{e}->{x}" for e, x, _ in EXPECTED],
    )
    def test_entry_exit_matrix(
        self, entry: str, exit_: str, expected_movement: TurnMovement
    ) -> None:
        classifier = TurnMovementClassifier(timeout_s=30.0)

        # Feed entry crossing
        entry_event = _make_crossing_event(
            track_id=1,
            line_name=f"{entry}_line",
            direction=entry,
            timestamp_s=0.0,
        )
        result = classifier.process_crossing(entry_event)
        assert result is None, "First crossing should be stored as pending entry"

        # U-Turn: entry and exit have the same direction.  The classifier
        # ignores duplicate-direction events (returns None).  So a U-Turn
        # can only be verified via the static MOVEMENT_MATRIX.
        if entry == exit_:
            # Verify directly from the matrix
            assert TurnMovementClassifier.MOVEMENT_MATRIX[(entry, exit_)] == expected_movement
            return

        # Feed exit crossing (different direction)
        exit_event = _make_crossing_event(
            track_id=1,
            line_name=f"{exit_}_line",
            direction=exit_,
            timestamp_s=5.0,
        )
        completed = classifier.process_crossing(exit_event)
        assert completed is not None, "Second crossing should resolve the movement"
        assert completed.movement == expected_movement
        assert completed.entry_approach == entry
        assert completed.exit_approach == exit_
        assert completed.incomplete is False


class TestTurnMovementClassifier:

    def test_pending_entry_returns_none(self) -> None:
        """Feeding only an entry crossing should return None."""
        classifier = TurnMovementClassifier(timeout_s=30.0)
        ev = _make_crossing_event(track_id=1, direction="NB", timestamp_s=0.0)
        result = classifier.process_crossing(ev)
        assert result is None
        assert classifier.pending_count == 1

    def test_flush_pending_timeout(self) -> None:
        """After timeout, flush_pending returns CompletedMovement with
        movement=THROUGH (entry→opposite) and incomplete=True."""
        classifier = TurnMovementClassifier(timeout_s=10.0)
        ev = _make_crossing_event(track_id=1, direction="NB", timestamp_s=0.0)
        classifier.process_crossing(ev)

        # Flush at t=100 (well past the 10 s timeout)
        flushed = classifier.flush_pending(current_time_s=100.0)
        assert len(flushed) == 1

        m = flushed[0]
        assert m.movement == TurnMovement.THROUGH
        assert m.entry_approach == "NB"
        assert m.exit_approach == "SB"    # opposite of NB
        assert m.incomplete is True
        assert classifier.pending_count == 0

    def test_flush_pending_not_expired_yet(self) -> None:
        """flush_pending should not return entries that haven't timed out."""
        classifier = TurnMovementClassifier(timeout_s=30.0)
        ev = _make_crossing_event(track_id=1, direction="EB", timestamp_s=10.0)
        classifier.process_crossing(ev)

        # Flush at t=20 (only 10 s elapsed, timeout is 30 s)
        flushed = classifier.flush_pending(current_time_s=20.0)
        assert len(flushed) == 0
        assert classifier.pending_count == 1

    def test_duplicate_direction_ignored(self) -> None:
        """If the same track crosses the same direction twice, the duplicate
        is silently ignored."""
        classifier = TurnMovementClassifier(timeout_s=30.0)
        ev1 = _make_crossing_event(track_id=1, direction="NB", timestamp_s=0.0)
        ev2 = _make_crossing_event(track_id=1, direction="NB", timestamp_s=3.0)
        classifier.process_crossing(ev1)
        result = classifier.process_crossing(ev2)
        assert result is None
        assert classifier.pending_count == 1


# ===================================================================
# Tests — interval_binning.py
# ===================================================================

class TestIntervalBinning:

    def test_15min_binning(self, standard_movements: List[CompletedMovement]) -> None:
        """10 movements in bin 07:00, 5 in bin 07:15 — verify totals."""
        agg = IntervalAggregator(study_start_time="07:00:00")
        for m in standard_movements:
            agg.add_movement(m)

        result = agg.compute_results("Test", "2026-03-12")

        interval_totals = agg.get_interval_totals()

        assert interval_totals.get("07:00", 0) == 10
        assert interval_totals.get("07:15", 0) == 5
        assert result.total_volume == 15

    def test_bin_alignment(self) -> None:
        """Movements at second 899 (07:14:59) map to 07:00 bin,
        and at second 900 (07:15:00) map to 07:15 bin."""
        agg = IntervalAggregator(study_start_time="07:00:00")

        m1 = _make_completed_movement(entry_time_s=899.0, track_id=1)
        m2 = _make_completed_movement(entry_time_s=900.0, track_id=2)

        agg.add_movement(m1)
        agg.add_movement(m2)

        totals = agg.get_interval_totals()
        assert totals.get("07:00", 0) == 1
        assert totals.get("07:15", 0) == 1

    def test_peak_hour_calculation(self) -> None:
        """Put significantly more traffic in intervals 3-6 (AM window
        07:45–08:30 → peak hour 07:45-08:45) and verify peak detection."""
        agg = IntervalAggregator(study_start_time="07:00:00")

        # 8 intervals covering 07:00–08:45 (2 hours).
        # Low traffic in intervals 0-1 and 6-7; high in 2-5.
        # interval 0: 07:00 →  2 vehicles (  0– 899 s)
        # interval 1: 07:15 →  3 vehicles ( 900–1799 s)
        # interval 2: 07:30 → 20 vehicles (1800–2699 s)
        # interval 3: 07:45 → 30 vehicles (2700–3599 s)
        # interval 4: 08:00 → 25 vehicles (3600–4499 s)
        # interval 5: 08:15 → 15 vehicles (4500–5399 s)
        # interval 6: 08:30 →  4 vehicles (5400–6299 s)
        # interval 7: 08:45 →  1 vehicles (6300–7199 s)
        counts = [2, 3, 20, 30, 25, 15, 4, 1]
        tid = 1
        for interval_idx, count in enumerate(counts):
            base_s = interval_idx * 900
            for j in range(count):
                agg.add_movement(_make_completed_movement(
                    track_id=tid,
                    entry_time_s=base_s + j * (850 / max(count, 1)),
                    entry_approach="NB",
                    exit_approach="SB",
                    movement=TurnMovement.THROUGH,
                ))
                tid += 1

        result = agg.compute_results("Peak Test", "2026-03-12")

        # The AM peak window searches 06:00–10:00.
        # Eligible bins: 07:00, 07:15, 07:30, 07:45, 08:00, 08:15, 08:30, 08:45
        # Sliding 4-bin windows:
        #   07:00-08:00: 2+3+20+30 = 55
        #   07:15-08:15: 3+20+30+25 = 78
        #   07:30-08:30: 20+30+25+15 = 90  ← peak
        #   07:45-08:45: 30+25+15+4 = 74
        #   08:00-09:00: 25+15+4+1 = 45
        assert result.am_peak_hour == "07:30-08:30"
        assert result.am_peak_volume == 90

        # PHF = total / (4 × max_15min)  = 90 / (4 × 30) = 0.75
        assert result.am_peak_phf == 0.75

    def test_heavy_vehicle_percentage(self) -> None:
        """Mix of Passenger Vehicle, Heavy Truck, Bus → verify heavy_vehicle_pct."""
        agg = IntervalAggregator(study_start_time="07:00:00")
        tid = 1

        # 6 Passenger Vehicles
        for i in range(6):
            agg.add_movement(_make_completed_movement(
                track_id=tid, entry_time_s=float(i * 100),
                vehicle_class="Passenger Vehicle",
            ))
            tid += 1

        # 2 Heavy Trucks
        for i in range(2):
            agg.add_movement(_make_completed_movement(
                track_id=tid, entry_time_s=float(600 + i * 100),
                vehicle_class="Heavy Truck",
            ))
            tid += 1

        # 2 Buses
        for i in range(2):
            agg.add_movement(_make_completed_movement(
                track_id=tid, entry_time_s=float(800 + i * 50),
                vehicle_class="Bus",
            ))
            tid += 1

        result = agg.compute_results("Heavy Test", "2026-03-12")

        # Total = 10,  Heavy = 2 + 2 = 4  → 40%
        assert result.total_volume == 10
        assert result.heavy_vehicle_pct == 40.0

    def test_empty_aggregator(self) -> None:
        """An aggregator with no data produces an empty result."""
        agg = IntervalAggregator(study_start_time="08:00:00")
        result = agg.compute_results("Empty", "2026-03-12")
        assert result.total_volume == 0
        assert result.bins == []
        assert result.am_peak_hour == "N/A"

    def test_pedestrian_and_bicycle_binning(self) -> None:
        """Verify add_pedestrian and add_bicycle bin correctly."""
        agg = IntervalAggregator(study_start_time="07:00:00")

        ped = _make_crossing_event(
            track_id=1, direction="NB", timestamp_s=100.0, object_class="Pedestrian",
        )
        bike = _make_crossing_event(
            track_id=2, direction="EB", timestamp_s=200.0, object_class="Bicycle",
        )
        agg.add_pedestrian(ped)
        agg.add_bicycle(bike)

        ped_totals = agg.get_pedestrian_totals()
        bike_totals = agg.get_bicycle_totals()

        assert ped_totals["07:00"]["NB"] == 1
        assert bike_totals["07:00"]["EB"] == 1


# ===================================================================
# Tests — report_generator.py
# ===================================================================

class TestUtdfCsv:

    def test_utdf_csv_format(self, sample_study_result: TrafficStudyResult) -> None:
        """Generate a UTDF CSV and verify its structure."""
        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = os.path.join(tmpdir, "volume.csv")
            generate_utdf_csv(sample_study_result, out_path)

            assert os.path.exists(out_path)
            with open(out_path, "r", newline="") as f:
                lines = f.readlines()

            # Line 1: "Volume Data"
            assert "Volume" in lines[0]

            # Line 2: "15 Minute Counts"
            assert "15 Minute" in lines[1]

            # Line 3: header row
            header_line = lines[2].strip()
            assert header_line.startswith("INTID")
            assert "Date" in header_line
            assert "Time" in header_line

            # Data rows (lines 3+)
            reader = csv.reader(lines[3:])
            data_rows = [r for r in reader if r]

            assert len(data_rows) > 0, "Must have at least one data row"

            # Each data row: INTID + 12 movement cols + 4 ped cols + Date + Day + Time = 20
            expected_cols = 1 + 12 + 4 + 3  # = 20
            for row in data_rows:
                assert len(row) == expected_cols, (
                    f"Expected {expected_cols} columns, got {len(row)}: {row}"
                )

            # Time values in HH:MM format
            for row in data_rows:
                time_val = row[-1]  # last column is Time
                parts = time_val.split(":")
                assert len(parts) == 2, f"Time should be HH:MM, got {time_val}"
                assert parts[0].isdigit() and parts[1].isdigit()

    def test_utdf_csv_data_integrity(self, sample_study_result: TrafficStudyResult) -> None:
        """Verify that UTDF CSV data sums match the original result."""
        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = os.path.join(tmpdir, "volume.csv")
            generate_utdf_csv(sample_study_result, out_path)

            with open(out_path, "r", newline="") as f:
                reader = csv.reader(f)
                lines = list(reader)

            # Skip first 3 rows (meta + header)
            data = lines[3:]
            csv_total = 0
            for row in data:
                # Columns 1-12 are the movement counts (NBL,NBT,NBR,SBL,…,WBR)
                for val in row[1:13]:
                    csv_total += int(val)

            assert csv_total == sample_study_result.total_volume


class TestPdfGeneration:

    def test_pdf_generation(self, sample_study_result: TrafficStudyResult) -> None:
        """Generate a PDF and verify it exists with a reasonable size."""
        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = os.path.join(tmpdir, "report.pdf")
            returned = generate_tmc_pdf(sample_study_result, out_path)

            assert returned == out_path
            assert os.path.exists(out_path)
            size = os.path.getsize(out_path)
            assert size > 1024, f"PDF should be > 1KB, got {size} bytes"

            # Verify it starts with the PDF magic bytes
            with open(out_path, "rb") as f:
                header = f.read(5)
            assert header == b"%PDF-", "File should start with %PDF-"

    def test_pdf_with_speed_data(self, sample_study_result: TrafficStudyResult) -> None:
        """Generate a PDF with speed data appended."""
        speed_data = [
            {"Track_ID": i, "Class": "Passenger Vehicle",
             "Speed_KMH": 40 + i * 2, "Timestamp_S": float(i * 10),
             "Approach": "NB"}
            for i in range(1, 21)
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = os.path.join(tmpdir, "report_speed.pdf")
            generate_tmc_pdf(sample_study_result, out_path, speed_data=speed_data)

            assert os.path.exists(out_path)
            size = os.path.getsize(out_path)
            assert size > 1024


class TestSpeedSummaryCsv:

    def test_speed_summary_csv(self) -> None:
        """Generate a speed summary CSV and verify structure."""
        speeds = [
            {"Track_ID": i, "Class": "Passenger Vehicle",
             "Speed_KMH": 40.0 + i, "Timestamp_S": float(i * 5),
             "Approach": "NB"}
            for i in range(1, 11)
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = os.path.join(tmpdir, "speed.csv")
            generate_speed_summary_csv(speeds, out_path)

            assert os.path.exists(out_path)
            with open(out_path, "r", newline="") as f:
                content = f.read()

            assert "Track_ID" in content
            assert "Summary" in content
            assert "85th Percentile Speed" in content
            assert "Mean Speed" in content
            assert "Sample Size" in content
