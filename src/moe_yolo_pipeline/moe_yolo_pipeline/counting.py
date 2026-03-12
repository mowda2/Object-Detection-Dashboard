"""
Counting Line Module for TrafficIQ

Provides virtual counting-line detection for tracked objects in video frames.
Users define one or more lines across a video frame; when a tracked object's
centroid crosses a line, a ``CrossingEvent`` is recorded with full metadata
(track ID, direction, timestamp, vehicle class, speed, bounding box).

Designed to integrate with the existing Supervision-based ByteTrack pipeline
used in ``speed_analyzer.py``.

Classes:
    CountingLine          – definition of a single virtual line
    CrossingEvent         – immutable record of one crossing
    LineCrossingDetector  – stateful detector that consumes per-frame
                            ``supervision.Detections`` and emits events

Functions:
    classify_vehicle      – maps COCO class IDs + bbox area to traffic-
                            engineering vehicle categories
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class CountingLine:
    """A virtual counting line defined by two pixel-coordinate endpoints.

    Attributes:
        name:      Human-readable identifier, e.g. ``"NB_approach"``.
        x1, y1:    Pixel coordinates of the line's start point.
        x2, y2:    Pixel coordinates of the line's end point.
        direction: Compass approach the line represents
                   (``"NB"``, ``"SB"``, ``"EB"``, ``"WB"``).
    """

    name: str
    x1: int
    y1: int
    x2: int
    y2: int
    direction: str  # "NB", "SB", "EB", "WB"

    def __post_init__(self) -> None:
        if self.direction not in {"NB", "SB", "EB", "WB"}:
            raise ValueError(
                f"direction must be one of NB, SB, EB, WB; got {self.direction!r}"
            )
        if self.x1 == self.x2 and self.y1 == self.y2:
            raise ValueError("Counting line start and end points must differ.")


@dataclass(frozen=True, slots=True)
class CrossingEvent:
    """Immutable record of a single line-crossing event.

    Attributes:
        track_id:     Persistent tracker ID assigned by ByteTrack.
        line_name:    Name of the ``CountingLine`` that was crossed.
        direction:    Compass approach of the crossed line.
        timestamp_s:  Seconds into the video when the crossing occurred.
        frame_number: 0-based frame index.
        object_class: Traffic-engineering class label (see
                      :func:`classify_vehicle`).
        speed_kmh:    Speed at time of crossing (``0.0`` if unavailable).
        bbox:         Bounding box ``(x1, y1, x2, y2)`` at the crossing frame.
    """

    track_id: int
    line_name: str
    direction: str
    timestamp_s: float
    frame_number: int
    object_class: str
    speed_kmh: float
    bbox: Tuple[float, float, float, float]


# ---------------------------------------------------------------------------
# Vehicle classification helper
# ---------------------------------------------------------------------------

#: Default bounding-box area threshold (px²) that separates light trucks from
#: heavy trucks when the COCO class is ``7`` (truck).
DEFAULT_TRUCK_AREA_THRESHOLD: float = 50_000.0

#: Mapping of COCO class IDs to traffic-engineering category labels.
_COCO_CLASS_MAP: Dict[int, str] = {
    0: "Pedestrian",
    1: "Bicycle",
    2: "Passenger Vehicle",
    3: "Motorcycle",
    5: "Bus",
    # 7 (truck) is handled separately to split light/heavy
}


def classify_vehicle(
    class_id: int,
    bbox_area: float,
    homography_scale: float = 1.0,
    truck_area_threshold: float = DEFAULT_TRUCK_AREA_THRESHOLD,
) -> str:
    """Map a COCO detection class ID to a traffic-engineering vehicle class.

    Parameters:
        class_id:             COCO class ID from the YOLO model.
        bbox_area:            Area of the bounding box in pixels² (width × height).
        homography_scale:     Optional multiplier applied to ``bbox_area`` before
                              comparing against the truck threshold.  Useful when a
                              homography is available to convert pixel area to
                              real-world area.  Defaults to ``1.0`` (no scaling).
        truck_area_threshold: Pixel-area threshold that separates *Light Truck*
                              from *Heavy Truck*.  Defaults to
                              :data:`DEFAULT_TRUCK_AREA_THRESHOLD` (50 000 px²).

    Returns:
        A human-readable vehicle-class string such as ``"Passenger Vehicle"``,
        ``"Heavy Truck"``, ``"Pedestrian"``, etc.

    Examples:
        >>> classify_vehicle(2, 12000)
        'Passenger Vehicle'
        >>> classify_vehicle(7, 30000)
        'Light Truck'
        >>> classify_vehicle(7, 80000)
        'Heavy Truck'
        >>> classify_vehicle(99, 500)
        'Other'
    """
    if class_id == 7:
        scaled_area = bbox_area * homography_scale
        return "Light Truck" if scaled_area < truck_area_threshold else "Heavy Truck"

    return _COCO_CLASS_MAP.get(class_id, "Other")


# ---------------------------------------------------------------------------
# Core crossing detector
# ---------------------------------------------------------------------------

#: Minimum pixel displacement past the line required before a crossing is
#: registered.  Prevents jitter / oscillation near the line from producing
#: duplicate counts.
_MIN_CROSSING_DISPLACEMENT: float = 5.0


class LineCrossingDetector:
    """Stateful detector that emits :class:`CrossingEvent` objects when tracked
    centroids cross any of the configured :class:`CountingLine` instances.

    The detector maintains the *previous centroid position* for every active
    track.  On each :meth:`update` call it computes the cross-product sign of
    the centroid with respect to each line; a sign change indicates a crossing.

    A minimum-displacement guard (``_MIN_CROSSING_DISPLACEMENT`` pixels past
    the line) prevents oscillation-induced double counts.

    Parameters:
        lines: One or more :class:`CountingLine` definitions.

    Raises:
        ValueError: If *lines* is empty.
    """

    def __init__(self, lines: List[CountingLine]) -> None:
        if not lines:
            raise ValueError("At least one CountingLine must be provided.")

        self._lines: List[CountingLine] = list(lines)

        # track_id -> (px, py) of centroid on the previous frame
        self._prev_centroids: Dict[int, Tuple[float, float]] = {}

        # track_id -> {line_name: last_sign}  (sign of cross-product)
        self._prev_signs: Dict[int, Dict[str, float]] = {}

        # track_id -> {line_name}  set of lines already crossed
        # (prevents the same track from being counted twice on the same line)
        self._crossed: Dict[int, set] = {}

    # ------------------------------------------------------------------
    # Geometry helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _point_side_of_line(px: float, py: float, line: CountingLine) -> float:
        """Return the cross-product sign of point ``(px, py)`` relative to
        the directed segment from ``(line.x1, line.y1)`` to
        ``(line.x2, line.y2)``.

        * **Positive** → point is to the *left* of the line direction.
        * **Negative** → point is to the *right*.
        * **Zero**     → point is exactly on the line.

        Parameters:
            px, py: Coordinates of the query point.
            line:   The counting line.

        Returns:
            A ``float`` whose sign encodes the side of the line.
        """
        return (
            (line.x2 - line.x1) * (py - line.y1)
            - (line.y2 - line.y1) * (px - line.x1)
        )

    @staticmethod
    def _distance_to_line(px: float, py: float, line: CountingLine) -> float:
        """Perpendicular distance from point ``(px, py)`` to *line*.

        Used to enforce the minimum-displacement guard.

        Parameters:
            px, py: Coordinates of the query point.
            line:   The counting line.

        Returns:
            Non-negative distance in pixels.
        """
        dx = line.x2 - line.x1
        dy = line.y2 - line.y1
        length = math.hypot(dx, dy)
        if length == 0.0:
            return math.hypot(px - line.x1, py - line.y1)
        return abs(dx * (py - line.y1) - dy * (px - line.x1)) / length

    # ------------------------------------------------------------------
    # Main update
    # ------------------------------------------------------------------

    def update(
        self,
        tracked_detections,
        frame_number: int,
        fps: float,
        class_names: Dict[int, str],
        speeds: Optional[Dict[int, float]] = None,
    ) -> List[CrossingEvent]:
        """Process one frame of tracked detections and return any new crossings.

        Parameters:
            tracked_detections:
                A ``supervision.Detections`` object produced by
                ``ByteTrack.update_with_detections``.  Must expose
                ``.tracker_id`` (``ndarray[int]``), ``.xyxy``
                (``ndarray[N,4]``), and ``.class_id`` (``ndarray[int]``).
            frame_number:
                0-based frame index of the current frame.
            fps:
                Frames per second of the source video (used to compute
                ``timestamp_s``).
            class_names:
                Mapping of COCO class IDs to short label strings (e.g.
                ``{2: "car", 7: "truck"}``).  Used for the ``object_class``
                field of :class:`CrossingEvent` via :func:`classify_vehicle`.
            speeds:
                Optional mapping of ``track_id → speed_kmh``.  When provided,
                the speed at the moment of crossing is recorded in the event.

        Returns:
            A list of :class:`CrossingEvent` instances generated this frame
            (may be empty).
        """
        events: List[CrossingEvent] = []

        if tracked_detections.tracker_id is None:
            return events

        speeds = speeds or {}
        timestamp_s = frame_number / fps if fps > 0 else 0.0

        active_ids: set = set()

        for det_idx in range(len(tracked_detections.tracker_id)):
            track_id = int(tracked_detections.tracker_id[det_idx])
            active_ids.add(track_id)

            x1, y1, x2, y2 = tracked_detections.xyxy[det_idx]
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0

            class_id = int(tracked_detections.class_id[det_idx])
            bbox_area = float((x2 - x1) * (y2 - y1))
            object_class = classify_vehicle(class_id, bbox_area)

            # Initialise sign bookkeeping for new tracks
            if track_id not in self._prev_signs:
                self._prev_signs[track_id] = {}
                self._crossed[track_id] = set()

            for line in self._lines:
                current_sign = self._point_side_of_line(cx, cy, line)

                prev_sign = self._prev_signs[track_id].get(line.name)

                if prev_sign is not None:
                    # Detect sign change (crossing)
                    crossed = (prev_sign > 0 and current_sign < 0) or (
                        prev_sign < 0 and current_sign > 0
                    )

                    if crossed and line.name not in self._crossed[track_id]:
                        # Minimum-displacement guard: the centroid must be at
                        # least _MIN_CROSSING_DISPLACEMENT pixels past the
                        # line to avoid jitter-triggered double counts.
                        dist = self._distance_to_line(cx, cy, line)
                        if dist >= _MIN_CROSSING_DISPLACEMENT:
                            speed_kmh = speeds.get(track_id, 0.0)
                            events.append(
                                CrossingEvent(
                                    track_id=track_id,
                                    line_name=line.name,
                                    direction=line.direction,
                                    timestamp_s=round(timestamp_s, 4),
                                    frame_number=frame_number,
                                    object_class=object_class,
                                    speed_kmh=round(speed_kmh, 2),
                                    bbox=(
                                        float(x1),
                                        float(y1),
                                        float(x2),
                                        float(y2),
                                    ),
                                )
                            )
                            self._crossed[track_id].add(line.name)

                # Always update the stored sign (even if no crossing yet)
                self._prev_signs[track_id][line.name] = current_sign

            # Store centroid for potential future use (e.g. trajectory export)
            self._prev_centroids[track_id] = (cx, cy)

        # Prune stale tracks that are no longer present in detections
        stale_ids = set(self._prev_centroids.keys()) - active_ids
        for stale_id in stale_ids:
            self._prev_centroids.pop(stale_id, None)
            self._prev_signs.pop(stale_id, None)
            self._crossed.pop(stale_id, None)

        return events

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Clear all internal state (centroids, signs, crossed sets)."""
        self._prev_centroids.clear()
        self._prev_signs.clear()
        self._crossed.clear()

    @property
    def lines(self) -> List[CountingLine]:
        """Return a copy of the configured counting lines."""
        return list(self._lines)

    def summary(self, events: List[CrossingEvent]) -> Dict[str, Dict[str, int]]:
        """Aggregate a list of events into per-line, per-class counts.

        Parameters:
            events: Crossing events (typically accumulated over a full video).

        Returns:
            Nested dict ``{line_name: {object_class: count}}``.

        Example::

            {
                "NB_approach": {"Passenger Vehicle": 42, "Heavy Truck": 3},
                "SB_approach": {"Bus": 1},
            }
        """
        counts: Dict[str, Dict[str, int]] = {}
        for ev in events:
            line_counts = counts.setdefault(ev.line_name, {})
            line_counts[ev.object_class] = line_counts.get(ev.object_class, 0) + 1
        return counts
