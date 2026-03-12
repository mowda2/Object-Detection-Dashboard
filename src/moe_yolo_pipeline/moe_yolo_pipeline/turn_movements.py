"""
Turn Movement Classification Module for TrafficIQ

Consumes :class:`~counting.CrossingEvent` objects produced by
:class:`~counting.LineCrossingDetector` and determines the turn movement each
vehicle made at the intersection: **Left**, **Through**, **Right**, or
**U-Turn**.

The logic assumes a standard four-leg intersection where each leg is labelled
with a compass approach (``NB``, ``SB``, ``EB``, ``WB``).  A vehicle that
*enters* on one approach and *exits* on another is classified via a static
movement matrix.

Classes:
    TurnMovement              – enum of possible turn types
    CompletedMovement         – immutable record of a fully classified
                                entry→exit movement
    TurnMovementClassifier    – stateful classifier; feed it crossing events
                                one at a time and collect completed movements

Typical usage::

    from counting import LineCrossingDetector, CrossingEvent
    from turn_movements import TurnMovementClassifier

    classifier = TurnMovementClassifier(timeout_s=30.0)
    all_movements: list[CompletedMovement] = []

    for frame in video:
        events = detector.update(...)
        for ev in events:
            result = classifier.process_crossing(ev)
            if result is not None:
                all_movements.append(result)

    # At end-of-video, flush any vehicles that entered but never exited
    all_movements.extend(classifier.flush_pending(video_duration_s))
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional

from .counting import CrossingEvent


# ---------------------------------------------------------------------------
# Enum
# ---------------------------------------------------------------------------

class TurnMovement(str, Enum):
    """Possible turn-movement types at a four-leg intersection.

    Inherits from ``str`` so that ``TurnMovement.LEFT == "L"`` is ``True`` and
    values serialise naturally to JSON.
    """

    LEFT = "L"
    THROUGH = "T"
    RIGHT = "R"
    UTURN = "U"
    UNKNOWN = "?"


# ---------------------------------------------------------------------------
# Data class
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class CompletedMovement:
    """Immutable record of a fully-classified entry→exit movement.

    Attributes:
        track_id:       Persistent tracker ID assigned by ByteTrack.
        entry_approach: Compass approach on which the vehicle entered
                        (``"NB"``, ``"SB"``, ``"EB"``, ``"WB"``).
        exit_approach:  Compass approach on which the vehicle exited.
        movement:       Classified turn movement (:class:`TurnMovement`).
        entry_time_s:   Timestamp (seconds into video) of the entry crossing.
        exit_time_s:    Timestamp (seconds into video) of the exit crossing.
        vehicle_class:  Traffic-engineering vehicle class (e.g.
                        ``"Passenger Vehicle"``).
        speed_kmh:      Speed at the *entry* crossing (``0.0`` if unavailable).
        incomplete:     ``True`` when the exit was *inferred* via
                        :meth:`TurnMovementClassifier.flush_pending` rather
                        than observed.  Defaults to ``False``.
    """

    track_id: int
    entry_approach: str
    exit_approach: str
    movement: TurnMovement
    entry_time_s: float
    exit_time_s: float
    vehicle_class: str
    speed_kmh: float
    incomplete: bool = False


# ---------------------------------------------------------------------------
# Classifier
# ---------------------------------------------------------------------------

#: Maps each entry approach to its *opposite* approach (used as the assumed
#: exit for timed-out through movements).
_OPPOSITE_APPROACH: Dict[str, str] = {
    "NB": "SB",
    "SB": "NB",
    "EB": "WB",
    "WB": "EB",
}


class TurnMovementClassifier:
    """Stateful classifier that pairs entry/exit crossing events and looks up
    the resulting turn movement.

    **Workflow**

    1. For each :class:`~counting.CrossingEvent` produced by the detector,
       call :meth:`process_crossing`.

       * If the track has *not* been seen before, the event is stored as a
         *pending entry*.
       * If the track *has* a pending entry **and** the new event's approach
         differs, the pair is resolved into a :class:`CompletedMovement` and
         returned.
       * If the approaches match (same line crossed twice), the duplicate is
         silently ignored to avoid double-counting.

    2. After all frames have been processed (or periodically), call
       :meth:`flush_pending` with the current video time to resolve any
       vehicles that entered but never exited within the timeout window.
       These are assumed to be *through* movements (entry→opposite approach)
       and are flagged with ``incomplete=True``.

    Parameters:
        timeout_s: Maximum elapsed seconds between entry and exit before a
                   pending entry is flushed as a through movement.
                   Defaults to ``30.0``.
    """

    #: Standard four-leg intersection movement matrix.
    #: ``(entry_approach, exit_approach) → TurnMovement``
    MOVEMENT_MATRIX: Dict[tuple[str, str], TurnMovement] = {
        # Northbound entry
        ("NB", "SB"): TurnMovement.THROUGH,
        ("NB", "EB"): TurnMovement.RIGHT,
        ("NB", "WB"): TurnMovement.LEFT,
        ("NB", "NB"): TurnMovement.UTURN,
        # Southbound entry
        ("SB", "NB"): TurnMovement.THROUGH,
        ("SB", "WB"): TurnMovement.RIGHT,
        ("SB", "EB"): TurnMovement.LEFT,
        ("SB", "SB"): TurnMovement.UTURN,
        # Eastbound entry
        ("EB", "WB"): TurnMovement.THROUGH,
        ("EB", "NB"): TurnMovement.LEFT,
        ("EB", "SB"): TurnMovement.RIGHT,
        ("EB", "EB"): TurnMovement.UTURN,
        # Westbound entry
        ("WB", "EB"): TurnMovement.THROUGH,
        ("WB", "SB"): TurnMovement.LEFT,
        ("WB", "NB"): TurnMovement.RIGHT,
        ("WB", "WB"): TurnMovement.UTURN,
    }

    def __init__(self, timeout_s: float = 30.0) -> None:
        if timeout_s <= 0:
            raise ValueError(f"timeout_s must be positive; got {timeout_s}")
        self._timeout_s = timeout_s

        # track_id → first (entry) CrossingEvent for vehicles that have
        # entered but not yet exited.
        self._pending: Dict[int, CrossingEvent] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process_crossing(self, event: CrossingEvent) -> Optional[CompletedMovement]:
        """Ingest a single crossing event and optionally return a completed
        movement.

        Parameters:
            event: A :class:`~counting.CrossingEvent` produced by
                   :class:`~counting.LineCrossingDetector`.

        Returns:
            A :class:`CompletedMovement` if an entry/exit pair was resolved,
            otherwise ``None``.
        """
        track_id = event.track_id
        entry = self._pending.get(track_id)

        if entry is None:
            # First time this track crosses any line → record as entry.
            self._pending[track_id] = event
            return None

        # Same approach as the stored entry → duplicate / jitter; ignore.
        if event.direction == entry.direction:
            return None

        # Different approach → this is the exit crossing.
        movement = self.MOVEMENT_MATRIX.get(
            (entry.direction, event.direction),
            TurnMovement.UNKNOWN,
        )

        completed = CompletedMovement(
            track_id=track_id,
            entry_approach=entry.direction,
            exit_approach=event.direction,
            movement=movement,
            entry_time_s=entry.timestamp_s,
            exit_time_s=event.timestamp_s,
            vehicle_class=entry.object_class,
            speed_kmh=entry.speed_kmh,
            incomplete=False,
        )

        del self._pending[track_id]
        return completed

    def flush_pending(self, current_time_s: float) -> List[CompletedMovement]:
        """Resolve all pending entries that have exceeded the timeout.

        Vehicles that entered but never exited within *timeout_s* are assumed
        to have travelled *through* the intersection (entry → opposite
        approach) and are returned with ``incomplete=True``.

        Parameters:
            current_time_s: The current timestamp in seconds (e.g. video
                            duration or current playback position).

        Returns:
            A list of :class:`CompletedMovement` instances for every timed-out
            entry (may be empty).
        """
        flushed: List[CompletedMovement] = []
        expired_ids: List[int] = []

        for track_id, entry in self._pending.items():
            if (current_time_s - entry.timestamp_s) >= self._timeout_s:
                exit_approach = _OPPOSITE_APPROACH.get(
                    entry.direction, entry.direction
                )
                movement = self.MOVEMENT_MATRIX.get(
                    (entry.direction, exit_approach),
                    TurnMovement.THROUGH,
                )

                flushed.append(
                    CompletedMovement(
                        track_id=track_id,
                        entry_approach=entry.direction,
                        exit_approach=exit_approach,
                        movement=movement,
                        entry_time_s=entry.timestamp_s,
                        exit_time_s=current_time_s,
                        vehicle_class=entry.object_class,
                        speed_kmh=entry.speed_kmh,
                        incomplete=True,
                    )
                )
                expired_ids.append(track_id)

        for tid in expired_ids:
            del self._pending[tid]

        return flushed

    def flush_all(self, current_time_s: float) -> List[CompletedMovement]:
        """Force-flush *every* pending entry regardless of timeout.

        Useful at end-of-video when you want to resolve all remaining tracks.
        All returned movements are marked ``incomplete=True``.

        Parameters:
            current_time_s: The current timestamp in seconds.

        Returns:
            A list of :class:`CompletedMovement` instances.
        """
        flushed: List[CompletedMovement] = []
        for track_id, entry in self._pending.items():
            exit_approach = _OPPOSITE_APPROACH.get(
                entry.direction, entry.direction
            )
            movement = self.MOVEMENT_MATRIX.get(
                (entry.direction, exit_approach),
                TurnMovement.THROUGH,
            )
            flushed.append(
                CompletedMovement(
                    track_id=track_id,
                    entry_approach=entry.direction,
                    exit_approach=exit_approach,
                    movement=movement,
                    entry_time_s=entry.timestamp_s,
                    exit_time_s=current_time_s,
                    vehicle_class=entry.object_class,
                    speed_kmh=entry.speed_kmh,
                    incomplete=True,
                )
            )
        self._pending.clear()
        return flushed

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Discard all pending entries without flushing."""
        self._pending.clear()

    @property
    def pending_count(self) -> int:
        """Number of vehicles that have entered but not yet exited."""
        return len(self._pending)

    @staticmethod
    def summary(
        movements: List[CompletedMovement],
    ) -> Dict[str, Dict[str, int]]:
        """Aggregate completed movements into a per-approach, per-turn count.

        Parameters:
            movements: Accumulated :class:`CompletedMovement` instances.

        Returns:
            Nested dict ``{entry_approach: {movement_value: count}}``.

        Example::

            {
                "NB": {"T": 38, "L": 12, "R": 9, "U": 1},
                "SB": {"T": 45, "L": 7,  "R": 11},
            }
        """
        counts: Dict[str, Dict[str, int]] = {}
        for m in movements:
            approach_counts = counts.setdefault(m.entry_approach, {})
            key = m.movement.value
            approach_counts[key] = approach_counts.get(key, 0) + 1
        return counts

    @staticmethod
    def detailed_summary(
        movements: List[CompletedMovement],
    ) -> Dict[str, Dict[str, Dict[str, int]]]:
        """Aggregate movements into per-approach, per-turn, per-vehicle-class
        counts.

        Parameters:
            movements: Accumulated :class:`CompletedMovement` instances.

        Returns:
            Nested dict
            ``{entry_approach: {movement_value: {vehicle_class: count}}}``.

        Example::

            {
                "NB": {
                    "T": {"Passenger Vehicle": 30, "Heavy Truck": 8},
                    "L": {"Passenger Vehicle": 12},
                },
            }
        """
        counts: Dict[str, Dict[str, Dict[str, int]]] = {}
        for m in movements:
            turn_map = counts.setdefault(m.entry_approach, {})
            class_map = turn_map.setdefault(m.movement.value, {})
            class_map[m.vehicle_class] = class_map.get(m.vehicle_class, 0) + 1
        return counts
