"""
Integration test: verify the full TrafficIQ pipeline connects correctly.
Simulates what speed_analyzer.py does when processing frames with counting lines.
"""

import os
import tempfile

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Helpers – lightweight mock that satisfies supervision.Detections interface
# ---------------------------------------------------------------------------


class _MockDetections:
    """Minimal stand-in for ``supervision.Detections`` as consumed by
    :meth:`LineCrossingDetector.update`.

    The detector accesses:
      * ``tracked_detections.tracker_id``  – ``ndarray[int]`` of shape ``(N,)``
      * ``tracked_detections.xyxy``        – ``ndarray[float]`` of shape ``(N, 4)``
      * ``tracked_detections.class_id``    – ``ndarray[int]`` of shape ``(N,)``
    """

    def __init__(self, tracker_id, xyxy, class_id):
        self.tracker_id = np.asarray(tracker_id, dtype=int) if tracker_id is not None else None
        self.xyxy = np.asarray(xyxy, dtype=float)
        self.class_id = np.asarray(class_id, dtype=int)


# ---------------------------------------------------------------------------
# Vehicle trajectory helpers
# ---------------------------------------------------------------------------

# For the intersection layout used in the test:
#
#    Frame is 640 × 480, intersection box is from (200,130) to (440,350).
#
#    NB line: y=350 (bottom of box),  vehicles enter from south going north
#    SB line: y=130 (top of box),     vehicles enter from north going south
#    EB line: x=200 (left of box),    vehicles enter from west going east
#    WB line: x=440 (right of box),   vehicles enter from east going west
#
# A "Northbound Through" vehicle enters by crossing the NB line (y=350)
# heading upward, then exits by crossing the SB line (y=130).
#
# The centroid must start clearly on one side of a line, then move clearly
# to the other side (>5 px displacement past the line).

def _through_nb_waypoints():
    """NB-Through: enters from bottom (NB line), exits at top (SB line)."""
    return [
        (320, 370),   # south of NB line (y=350)
        (320, 340),   # crossed NB line heading north
        (320, 240),   # middle of intersection
        (320, 120),   # crossed SB line heading north → exit
    ]


def _through_sb_waypoints():
    """SB-Through: enters from top (SB line), exits at bottom (NB line)."""
    return [
        (320, 110),   # north of SB line (y=130)
        (320, 140),   # crossed SB line heading south
        (320, 240),   # middle
        (320, 360),   # crossed NB line → exit
    ]


def _left_nb_waypoints():
    """NB-Left: enters NB line, exits WB line (turning left)."""
    return [
        (320, 370),   # south of NB line
        (320, 340),   # crossed NB line heading north
        (380, 260),   # curving toward WB (right side)
        (450, 240),   # crossed WB line (x=440) → exit
    ]


def _right_nb_waypoints():
    """NB-Right: enters NB line, exits EB line (turning right)."""
    return [
        (320, 370),   # south of NB line
        (320, 340),   # crossed NB line heading north
        (260, 260),   # curving toward EB (left side)
        (190, 240),   # crossed EB line (x=200) → exit
    ]


def _through_eb_waypoints():
    """EB-Through: enters EB line (x=200), exits WB line (x=440)."""
    return [
        (180, 240),   # west of EB line (x=200)
        (210, 240),   # crossed EB line heading east
        (320, 240),   # middle
        (450, 240),   # crossed WB line → exit
    ]


def _through_wb_waypoints():
    """WB-Through: enters WB line (x=440), exits EB line (x=200)."""
    return [
        (460, 240),   # east of WB line (x=440)
        (430, 240),   # crossed WB line heading west
        (320, 240),   # middle
        (190, 240),   # crossed EB line → exit
    ]


# ---------------------------------------------------------------------------
# Build per-frame detections from waypoint sequences
# ---------------------------------------------------------------------------

_BBOX_HALF = 30  # half-width/height of bounding box around centroid


def _make_frame_detections(vehicles_by_frame):
    """Given a dict ``{frame_idx: [(track_id, cx, cy, class_id), ...]}``,
    return a dict ``{frame_idx: _MockDetections}``."""
    frames = {}
    for frame_idx, vlist in vehicles_by_frame.items():
        if not vlist:
            continue
        tracker_ids = [v[0] for v in vlist]
        xyxy = []
        class_ids = []
        for _, cx, cy, cid in vlist:
            xyxy.append([
                cx - _BBOX_HALF, cy - _BBOX_HALF,
                cx + _BBOX_HALF, cy + _BBOX_HALF,
            ])
            class_ids.append(cid)
        frames[frame_idx] = _MockDetections(
            tracker_id=tracker_ids,
            xyxy=xyxy,
            class_id=class_ids,
        )
    return frames


# ---------------------------------------------------------------------------
# Main integration test
# ---------------------------------------------------------------------------

def test_full_pipeline_integration():
    """
    Simulate the per-frame processing loop:
    1. Create counting lines for a 4-leg intersection
    2. Create fake tracked detections that cross the lines
    3. Run them through LineCrossingDetector → TurnMovementClassifier → IntervalAggregator
    4. Generate reports from the result
    5. Verify everything connects without errors
    """
    from moe_yolo_pipeline.moe_yolo_pipeline.counting import (
        CountingLine, LineCrossingDetector, classify_vehicle,
    )
    from moe_yolo_pipeline.moe_yolo_pipeline.turn_movements import (
        TurnMovementClassifier, TurnMovement,
    )
    from moe_yolo_pipeline.moe_yolo_pipeline.interval_binning import (
        IntervalAggregator,
    )
    from moe_yolo_pipeline.moe_yolo_pipeline.report_generator import (
        generate_tmc_pdf, generate_utdf_csv,
    )

    # ── Step 1: Define counting lines ──────────────────────────────────
    lines = [
        CountingLine(name="NB", x1=200, y1=350, x2=440, y2=350, direction="NB"),
        CountingLine(name="SB", x1=200, y1=130, x2=440, y2=130, direction="SB"),
        CountingLine(name="EB", x1=200, y1=130, x2=200, y2=350, direction="EB"),
        CountingLine(name="WB", x1=440, y1=130, x2=440, y2=350, direction="WB"),
    ]

    detector = LineCrossingDetector(lines)
    classifier = TurnMovementClassifier(timeout_s=60.0)
    # Study starts at 07:00 — vehicles will appear in the 07:00-07:30 window
    aggregator = IntervalAggregator(study_start_time="07:00:00")

    # ── Step 2: Build vehicle trajectories ─────────────────────────────
    # We create ~20 vehicles making various turns, spaced across ~30 minutes
    # of video time. At 10 fps, 30 min = 18000 frames.
    FPS = 10.0
    COCO_CAR = 2
    COCO_TRUCK = 7

    # Each vehicle: (track_id, waypoints_fn, class_id, start_frame)
    # Spacing: 3 seconds (30 frames) between waypoint steps.
    FRAME_STEP = 30  # frames between waypoint positions

    vehicle_specs = [
        # NB through vehicles (cars)
        (1,  _through_nb_waypoints,  COCO_CAR,   0),
        (2,  _through_nb_waypoints,  COCO_CAR,   100),
        (3,  _through_nb_waypoints,  COCO_CAR,   200),
        (4,  _through_nb_waypoints,  COCO_CAR,   300),
        (5,  _through_nb_waypoints,  COCO_TRUCK, 400),
        # NB left turns
        (6,  _left_nb_waypoints,     COCO_CAR,   500),
        (7,  _left_nb_waypoints,     COCO_CAR,   600),
        # NB right turns
        (8,  _right_nb_waypoints,    COCO_CAR,   700),
        (9,  _right_nb_waypoints,    COCO_CAR,   800),
        # SB through vehicles
        (10, _through_sb_waypoints,  COCO_CAR,   1000),
        (11, _through_sb_waypoints,  COCO_CAR,   1100),
        (12, _through_sb_waypoints,  COCO_TRUCK, 1200),
        # EB through vehicles
        (13, _through_eb_waypoints,  COCO_CAR,   2000),
        (14, _through_eb_waypoints,  COCO_CAR,   2100),
        (15, _through_eb_waypoints,  COCO_CAR,   2200),
        # WB through vehicles
        (16, _through_wb_waypoints,  COCO_CAR,   3000),
        (17, _through_wb_waypoints,  COCO_CAR,   3100),
        # More NB through in the 07:15 bin (after frame 9000 = 900s = 15min)
        (18, _through_nb_waypoints,  COCO_CAR,   9100),
        (19, _through_nb_waypoints,  COCO_CAR,   9200),
        (20, _through_nb_waypoints,  COCO_CAR,   9300),
    ]

    # Build per-frame vehicle positions
    vehicles_by_frame = {}
    for track_id, waypoints_fn, class_id, start_frame in vehicle_specs:
        waypoints = waypoints_fn()
        for step_idx, (cx, cy) in enumerate(waypoints):
            frame = start_frame + step_idx * FRAME_STEP
            vlist = vehicles_by_frame.setdefault(frame, [])
            vlist.append((track_id, cx, cy, class_id))

    frame_detections = _make_frame_detections(vehicles_by_frame)

    # ── Step 3: Run detections through the full pipeline ───────────────
    all_events = []
    completed_movements = []

    class_names = {0: "person", 1: "bicycle", 2: "car", 3: "motorcycle",
                   5: "bus", 7: "truck"}

    for frame_idx in sorted(frame_detections.keys()):
        detections = frame_detections[frame_idx]
        events = detector.update(
            tracked_detections=detections,
            frame_number=frame_idx,
            fps=FPS,
            class_names=class_names,
            speeds=None,
        )
        all_events.extend(events)

        for ev in events:
            result = classifier.process_crossing(ev)
            if result is not None:
                completed_movements.append(result)

    # Flush any vehicles that entered but never exited
    video_duration_s = max(frame_detections.keys()) / FPS
    flushed = classifier.flush_all(video_duration_s)
    completed_movements.extend(flushed)

    # ── Step 3b: Validate crossing detection ───────────────────────────
    assert len(all_events) > 0, "No crossing events detected at all"
    print(f"  Crossing events detected: {len(all_events)}")
    print(f"  Completed movements: {len(completed_movements)}")

    # Each vehicle that traverses two lines should produce two crossing events
    # (entry + exit). Some may only produce one (flushed as incomplete).
    assert len(completed_movements) > 0, "No completed movements resolved"

    # Check that we got a mix of turn types
    turn_types_seen = {m.movement for m in completed_movements}
    print(f"  Turn types observed: {turn_types_seen}")
    assert TurnMovement.THROUGH in turn_types_seen, "Expected THROUGH movements"

    # ── Step 4: Add movements to aggregator and compute results ────────
    for m in completed_movements:
        aggregator.add_movement(m)

    result = aggregator.compute_results("Test & Main St", "2026-03-12")

    # Basic result validation
    assert result.total_volume > 0, "Total volume should be > 0"
    assert result.intersection_name == "Test & Main St"
    assert result.study_date == "2026-03-12"
    assert len(result.bins) > 0, "Should have at least one bin"

    # Check heavy vehicle percentage makes sense (we put in some trucks)
    # Trucks with small bbox area → "Light Truck" (classify_vehicle logic)
    print(f"  Total volume: {result.total_volume}")
    print(f"  Heavy vehicle %: {result.heavy_vehicle_pct}")
    print(f"  AM peak: {result.am_peak_hour} ({result.am_peak_volume} veh)")
    print(f"  PM peak: {result.pm_peak_hour} ({result.pm_peak_volume} veh)")

    # ── Step 5: Generate reports ───────────────────────────────────────
    with tempfile.TemporaryDirectory() as tmpdir:
        pdf_path = os.path.join(tmpdir, "tmc_report.pdf")
        csv_path = os.path.join(tmpdir, "utdf_volume.csv")

        generate_tmc_pdf(result, pdf_path)
        generate_utdf_csv(result, csv_path)

        assert os.path.exists(pdf_path), "PDF not generated"
        assert os.path.exists(csv_path), "CSV not generated"
        assert os.path.getsize(pdf_path) > 500, "PDF too small — likely empty"

        with open(csv_path) as f:
            csv_lines = f.readlines()
            assert len(csv_lines) > 3, "CSV has no data rows"
            # Verify UTDF header structure
            assert "Volume Data" in csv_lines[0]
            assert "15 Minute Counts" in csv_lines[1]
            assert "INTID" in csv_lines[2]

        print(f"  PDF size: {os.path.getsize(pdf_path):,} bytes")
        print(f"  CSV rows: {len(csv_lines)} (including 3 header lines)")

    print(f"\n✓ Integration test passed: {result.total_volume} vehicles processed")


# ---------------------------------------------------------------------------
# Focused sub-tests for each pipeline stage
# ---------------------------------------------------------------------------

def test_crossing_detector_with_mock_detections():
    """Verify that the LineCrossingDetector correctly fires events when a
    tracked centroid crosses from one side of a line to the other."""
    from moe_yolo_pipeline.moe_yolo_pipeline.counting import (
        CountingLine, LineCrossingDetector,
    )

    line = CountingLine(name="NB", x1=0, y1=200, x2=400, y2=200, direction="NB")
    detector = LineCrossingDetector([line])
    class_names = {2: "car"}

    # Frame 0: vehicle at (200, 220) — below the line
    det0 = _MockDetections(
        tracker_id=[1],
        xyxy=[[170, 190, 230, 250]],  # centroid = (200, 220)
        class_id=[2],
    )
    events = detector.update(det0, frame_number=0, fps=10.0, class_names=class_names)
    assert len(events) == 0, "Should not fire on the first frame (no history)"

    # Frame 1: vehicle at (200, 180) — above the line (crossed!)
    det1 = _MockDetections(
        tracker_id=[1],
        xyxy=[[170, 150, 230, 210]],  # centroid = (200, 180)
        class_id=[2],
    )
    events = detector.update(det1, frame_number=1, fps=10.0, class_names=class_names)
    assert len(events) == 1, f"Expected exactly 1 crossing event, got {len(events)}"
    assert events[0].track_id == 1
    assert events[0].direction == "NB"
    assert events[0].object_class == "Passenger Vehicle"


def test_turn_movement_classification():
    """Verify that entry→exit crossing pairs produce the correct turn movement."""
    from moe_yolo_pipeline.moe_yolo_pipeline.counting import CrossingEvent
    from moe_yolo_pipeline.moe_yolo_pipeline.turn_movements import (
        TurnMovementClassifier, TurnMovement,
    )

    classifier = TurnMovementClassifier(timeout_s=60.0)

    # NB entry
    entry = CrossingEvent(
        track_id=1, line_name="NB", direction="NB",
        timestamp_s=10.0, frame_number=100,
        object_class="Passenger Vehicle", speed_kmh=35.0,
        bbox=(100, 330, 160, 370),
    )
    result = classifier.process_crossing(entry)
    assert result is None, "First crossing should be stored as pending entry"

    # SB exit → NB-Through
    exit_ev = CrossingEvent(
        track_id=1, line_name="SB", direction="SB",
        timestamp_s=15.0, frame_number=150,
        object_class="Passenger Vehicle", speed_kmh=40.0,
        bbox=(100, 110, 160, 150),
    )
    result = classifier.process_crossing(exit_ev)
    assert result is not None, "Second crossing should complete the movement"
    assert result.movement == TurnMovement.THROUGH
    assert result.entry_approach == "NB"
    assert result.exit_approach == "SB"
    assert result.vehicle_class == "Passenger Vehicle"


def test_interval_aggregation():
    """Verify that completed movements are correctly binned into 15-min intervals."""
    from moe_yolo_pipeline.moe_yolo_pipeline.turn_movements import (
        CompletedMovement, TurnMovement,
    )
    from moe_yolo_pipeline.moe_yolo_pipeline.interval_binning import (
        IntervalAggregator,
    )

    agg = IntervalAggregator(study_start_time="07:00:00")

    # Movement at video second 120 → clock time 07:02 → bin "07:00"
    m1 = CompletedMovement(
        track_id=1, entry_approach="NB", exit_approach="SB",
        movement=TurnMovement.THROUGH,
        entry_time_s=120.0, exit_time_s=125.0,
        vehicle_class="Passenger Vehicle", speed_kmh=35.0,
    )
    agg.add_movement(m1)

    # Movement at video second 1020 → clock time 07:17 → bin "07:15"
    m2 = CompletedMovement(
        track_id=2, entry_approach="NB", exit_approach="SB",
        movement=TurnMovement.THROUGH,
        entry_time_s=1020.0, exit_time_s=1025.0,
        vehicle_class="Heavy Truck", speed_kmh=30.0,
    )
    agg.add_movement(m2)

    result = agg.compute_results("Test Intersection", "2026-03-12")

    assert result.total_volume == 2
    assert result.heavy_vehicle_pct > 0  # one truck out of two
    assert len(result.bins) > 0

    # Verify the bins cover the expected intervals
    bin_labels = {b.interval_start for b in result.bins}
    assert "07:00" in bin_labels
    assert "07:15" in bin_labels


def test_classify_vehicle():
    """Verify COCO → traffic-engineering class mapping."""
    from moe_yolo_pipeline.moe_yolo_pipeline.counting import classify_vehicle

    assert classify_vehicle(2, 5000) == "Passenger Vehicle"
    assert classify_vehicle(7, 30000) == "Light Truck"
    assert classify_vehicle(7, 80000) == "Heavy Truck"
    assert classify_vehicle(5, 10000) == "Bus"
    assert classify_vehicle(3, 2000) == "Motorcycle"
    assert classify_vehicle(0, 1000) == "Pedestrian"
    assert classify_vehicle(1, 1500) == "Bicycle"
    assert classify_vehicle(99, 500) == "Other"
