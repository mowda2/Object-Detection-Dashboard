"""
Speed Analyzer Module - PASS B (v2 with improvements)

Implements homography-based speed measurement for tracked objects.
This module is completely isolated from offline_analyzer.py.

Main function:
    run_speed_job(job_dir: str, settings: dict = None) -> dict

Improvements (v2):
- Configurable settings via settings.json
- Improved filtering: min_box_area, ROI, aspect_ratio
- Outlier rejection for speed spikes
- Median + EMA smoothing
- Frame skip support
- Optional resize for faster inference
- Cache fingerprinting for reproducible reruns
"""

import os
import json
import csv
import time
import hashlib
import shutil
import subprocess
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict, deque
from statistics import median

import cv2
import numpy as np
from ultralytics import YOLO
import supervision as sv


# ---------------------------------------------------------------------------
# Video Codec Utilities (for browser compatibility)
# ---------------------------------------------------------------------------

def reencode_video_for_browser(input_path: str, output_path: str = None) -> bool:
    """
    Re-encode video to H.264 for browser compatibility (Chrome, Firefox, Edge).
    Uses ffmpeg if available. Falls back to original if ffmpeg not found.
    
    Args:
        input_path: Path to input video (mp4v codec)
        output_path: Path for output video. If None, replaces input in-place.
    
    Returns:
        True if re-encoding succeeded, False otherwise
    """
    if output_path is None:
        output_path = input_path
    
    # Check if ffmpeg is available
    ffmpeg_path = shutil.which("ffmpeg")
    if not ffmpeg_path:
        return False
    
    try:
        # Create temp output path
        temp_output = input_path + ".h264.mp4"
        
        # Re-encode with H.264 (libx264) for browser compatibility
        # -y: overwrite output
        # -i: input file
        # -c:v libx264: use H.264 codec
        # -preset fast: balance speed/quality
        # -crf 23: quality (lower = better, 23 is default)
        # -pix_fmt yuv420p: pixel format for compatibility
        # -movflags +faststart: optimize for web streaming
        cmd = [
            ffmpeg_path, "-y", "-i", input_path,
            "-c:v", "libx264", "-preset", "fast", "-crf", "23",
            "-pix_fmt", "yuv420p", "-movflags", "+faststart",
            "-an",  # no audio (our videos don't have audio)
            temp_output
        ]
        
        result = subprocess.run(cmd, capture_output=True, timeout=300)
        
        if result.returncode == 0 and os.path.exists(temp_output):
            # Replace original with re-encoded version
            os.replace(temp_output, output_path)
            return True
        else:
            # Clean up failed temp file
            if os.path.exists(temp_output):
                os.remove(temp_output)
            return False
            
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Environment variable configuration (isolated from offline mode)
# ---------------------------------------------------------------------------

def _env_float(name: str, default: float) -> float:
    """Read a float from env var."""
    val = os.environ.get(name, "")
    if not val:
        return default
    try:
        return float(val)
    except ValueError:
        return default


def _env_int(name: str, default: int) -> int:
    """Read an int from env var."""
    val = os.environ.get(name, "")
    if not val:
        return default
    try:
        return int(val)
    except ValueError:
        return default


def _env_str(name: str, default: str) -> str:
    """Read a string from env var."""
    return os.environ.get(name, default)


# Speed-specific configuration (env vars as fallback defaults)
MOE_SPEED_MAX_KPH = lambda: _env_float("MOE_SPEED_MAX_KPH", 160.0)
MOE_SPEED_WINDOW_S = lambda: _env_float("MOE_SPEED_WINDOW_S", 0.7)
MOE_SPEED_MIN_DT_S = lambda: _env_float("MOE_SPEED_MIN_DT_S", 0.1)
MOE_SPEED_CLASSES = lambda: _env_str("MOE_SPEED_CLASSES", "")
MOE_SPEED_CONF = lambda: _env_float("MOE_SPEED_CONF", 0.25)


# ---------------------------------------------------------------------------
# Default Settings (used when no settings.json provided)
# ---------------------------------------------------------------------------

DEFAULT_SETTINGS = {
    # Detection settings
    "min_conf": 0.25,
    "max_det_per_frame": 50,
    "class_filter": "",  # comma-separated class IDs, empty = all
    "min_box_area": 500,  # minimum bbox area in pixels^2
    "aspect_ratio_filter": False,  # enable aspect ratio filtering
    "aspect_ratio_min": 0.3,  # min width/height ratio (for vehicles)
    "aspect_ratio_max": 4.0,  # max width/height ratio
    
    # ROI filter (disabled by default)
    "roi_enabled": False,
    "roi_x1": 0,
    "roi_y1": 0,
    "roi_x2": 0,
    "roi_y2": 0,
    
    # Tracking (ByteTrack) settings
    "track_thresh": 0.25,  # track_activation_threshold
    "match_thresh": 0.8,   # minimum_matching_threshold
    "track_buffer": 30,    # lost_track_buffer
    
    # Speed calculation settings
    "smoothing_window_s": 0.7,
    "max_kph": 160.0,
    "min_dt_s": 0.1,
    "outlier_reject_kph_per_s": 50.0,  # max acceleration (km/h per second)
    "median_filter_n": 5,  # median filter window (0 = disabled)
    "jump_reject_m": 15.0,  # reject world position jumps > this (meters)
    
    # Performance settings
    "frame_skip": 1,  # process every Nth frame
    "resize_width": 0,  # 0 = disabled; e.g., 640 for faster inference
    
    # Trail visualization settings (motion trails behind tracked objects)
    "enable_trails": False,  # disabled by default for backward compatibility
    "trail_seconds": 2.0,    # time-based history window in seconds
    "trail_max_points": 60,  # safety cap on history points per track
    "trail_stride": 1,       # keep every Nth point (1 = all points)
    "trail_thickness": 2,    # line thickness in pixels
    "trail_alpha": 0.85,     # overall intensity for fading overlay
    "trail_fade": True,      # older segments fade out
    "trail_anchor": "bottom_center",  # "bottom_center" or "center"
    
    # Violation detection settings (speed violations capture)
    "violation_enabled": False,           # disabled by default
    "violation_speed_kmh": 110.0,         # speed threshold for violation
    "violation_min_seconds": 0.3,         # must be above threshold for this long
    "violation_cooldown_seconds": 2.0,    # prevent repeated snapshots for same track
    "violation_capture_mode": "peak_speed",  # "first_crossing" or "peak_speed"
    "violation_save_full_frame": True,    # save full frame snapshot
    "violation_save_crop": True,          # save cropped bbox snapshot
    "violation_crop_padding_px": 20,      # padding around bbox for crop
}


def get_default_settings() -> dict:
    """Return a copy of default settings with env var overrides."""
    settings = DEFAULT_SETTINGS.copy()
    # Override with env vars if set
    settings["min_conf"] = MOE_SPEED_CONF()
    settings["max_kph"] = MOE_SPEED_MAX_KPH()
    settings["smoothing_window_s"] = MOE_SPEED_WINDOW_S()
    settings["min_dt_s"] = MOE_SPEED_MIN_DT_S()
    class_filter = MOE_SPEED_CLASSES()
    if class_filter:
        settings["class_filter"] = class_filter
    return settings


def _settings_hash(settings: dict) -> str:
    """Compute a hash of settings for cache fingerprinting."""
    # Include all settings that affect output (analysis + visualization)
    keys_for_hash = [
        "min_conf", "max_det_per_frame", "class_filter", "min_box_area",
        "aspect_ratio_filter", "aspect_ratio_min", "aspect_ratio_max",
        "roi_enabled", "roi_x1", "roi_y1", "roi_x2", "roi_y2",
        "track_thresh", "match_thresh", "track_buffer",
        "smoothing_window_s", "max_kph", "min_dt_s", "outlier_reject_kph_per_s",
        "median_filter_n", "jump_reject_m", "frame_skip", "resize_width",
        # Trail settings (affect video output)
        "enable_trails", "trail_seconds", "trail_max_points", "trail_stride",
        "trail_thickness", "trail_alpha", "trail_fade", "trail_anchor",
        # Violation settings (affect video output and violation artifacts)
        "violation_enabled", "violation_speed_kmh", "violation_min_seconds",
        "violation_cooldown_seconds", "violation_capture_mode",
        "violation_save_full_frame", "violation_save_crop", "violation_crop_padding_px",
    ]
    subset = {k: settings.get(k) for k in keys_for_hash}
    s = json.dumps(subset, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(s.encode()).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Track Statistics for Speed Analysis
# ---------------------------------------------------------------------------

class SpeedTrackStats:
    """
    Collects per-track statistics including speed measurements.
    """
    
    def __init__(self):
        self._tracks: Dict[int, dict] = {}
    
    def update(self, track_id: int, frame_idx: int, class_name: str,
               speed_kph: float, rejected: bool = False):
        """Update stats for a track."""
        if track_id not in self._tracks:
            self._tracks[track_id] = {
                "track_id": track_id,
                "class": class_name,
                "start_frame": frame_idx,
                "end_frame": frame_idx,
                "frames_seen": 1,
                "speeds_kph": [speed_kph] if speed_kph > 0 else [],
                "max_kph": speed_kph,
                "rejected_updates": 0,
            }
        else:
            s = self._tracks[track_id]
            s["end_frame"] = frame_idx
            s["frames_seen"] += 1
            if rejected:
                s["rejected_updates"] += 1
            elif speed_kph > 0:
                s["speeds_kph"].append(speed_kph)
                if speed_kph > s["max_kph"]:
                    s["max_kph"] = speed_kph
    
    def get_summary(self) -> Dict[int, dict]:
        """Get summary dict for all tracks."""
        result = {}
        for tid, s in self._tracks.items():
            speeds = s["speeds_kph"]
            avg_kph = sum(speeds) / len(speeds) if speeds else 0.0
            result[tid] = {
                "avg_kph": round(avg_kph, 2),
                "max_kph": round(s["max_kph"], 2),
                "frames_seen": s["frames_seen"],
                "start_frame": s["start_frame"],
                "end_frame": s["end_frame"],
                "class": s["class"],
                "rejected_updates": s.get("rejected_updates", 0),
            }
        return result
    
    @property
    def unique_tracks(self) -> int:
        return len(self._tracks)


# ---------------------------------------------------------------------------
# Speed Calculator with Smoothing and Outlier Rejection
# ---------------------------------------------------------------------------

class SpeedCalculator:
    """
    Calculates speed from world coordinates using a sliding window.
    Maintains per-track history for smoothing.
    
    Improvements (v2):
    - Outlier rejection based on max acceleration
    - Median filter for robust smoothing
    - Jump detection for implausible position changes
    """
    
    def __init__(self, window_s: float = 0.7, min_dt_s: float = 0.1, 
                 max_kph: float = 160.0, outlier_reject_kph_per_s: float = 50.0,
                 median_filter_n: int = 5, jump_reject_m: float = 15.0):
        self.window_s = window_s
        self.min_dt_s = min_dt_s
        self.max_kph = max_kph
        self.outlier_reject_kph_per_s = outlier_reject_kph_per_s
        self.median_filter_n = median_filter_n
        self.jump_reject_m = jump_reject_m
        
        # Per-track history: deque of (t_seconds, X_m, Y_m)
        self._history: Dict[int, deque] = defaultdict(lambda: deque(maxlen=100))
        # Per-track smoothed speed
        self._smoothed: Dict[int, float] = {}
        # Per-track recent raw speeds for median filter
        self._recent_speeds: Dict[int, deque] = defaultdict(lambda: deque(maxlen=20))
        # Per-track last valid position for jump detection
        self._last_valid_pos: Dict[int, Tuple[float, float, float]] = {}  # (t, x, y)
    
    def update(self, track_id: int, t_s: float, x_m: float, y_m: float) -> Tuple[float, float, bool]:
        """
        Update track position and compute speed.
        
        Returns:
            (speed_mps, speed_kph, was_rejected) - speeds are 0 if not enough data
            was_rejected is True if this update was rejected as an outlier
        """
        hist = self._history[track_id]
        
        # Jump detection: check if position jumped implausibly
        if track_id in self._last_valid_pos:
            last_t, last_x, last_y = self._last_valid_pos[track_id]
            dt = t_s - last_t
            if dt > 0:
                jump_dist = np.sqrt((x_m - last_x)**2 + (y_m - last_y)**2)
                # Max plausible distance = (max_kph / 3.6) * dt * 1.5 (with margin)
                max_dist = (self.max_kph / 3.6) * dt * 1.5
                if jump_dist > max(self.jump_reject_m, max_dist):
                    # Position jump too large, reject this update
                    prev = self._smoothed.get(track_id, 0.0)
                    return prev / 3.6, prev, True
        
        # Add to history
        hist.append((t_s, x_m, y_m))
        self._last_valid_pos[track_id] = (t_s, x_m, y_m)
        
        if len(hist) < 2:
            return 0.0, 0.0, False
        
        # Find oldest point within window
        current_t, current_x, current_y = hist[-1]
        oldest_t, oldest_x, oldest_y = hist[0]
        
        # Try to find a point at least window_s ago
        for pt in hist:
            if current_t - pt[0] >= self.window_s:
                oldest_t, oldest_x, oldest_y = pt
                break
        
        dt = current_t - oldest_t
        if dt < self.min_dt_s:
            # Not enough time elapsed, use previous smoothed value
            prev = self._smoothed.get(track_id, 0.0)
            return prev / 3.6, prev, False
        
        # Compute distance
        dx = current_x - oldest_x
        dy = current_y - oldest_y
        dist_m = np.sqrt(dx*dx + dy*dy)
        
        # Speed in m/s and km/h
        speed_mps = dist_m / dt
        speed_kph = speed_mps * 3.6
        
        # Outlier rejection: check acceleration
        prev_kph = self._smoothed.get(track_id, speed_kph)
        accel = abs(speed_kph - prev_kph) / max(dt, 0.001)  # km/h per second
        
        if accel > self.outlier_reject_kph_per_s and prev_kph > 0:
            # Acceleration too high, likely ID switch or tracking error
            # Return previous value but mark as rejected
            return prev_kph / 3.6, prev_kph, True
        
        # Clamp to max
        if speed_kph > self.max_kph:
            speed_kph = self.max_kph
            speed_mps = speed_kph / 3.6
        
        # Add to recent speeds for median filter
        self._recent_speeds[track_id].append(speed_kph)
        
        # Apply median filter if enabled
        if self.median_filter_n > 0 and len(self._recent_speeds[track_id]) >= self.median_filter_n:
            recent = list(self._recent_speeds[track_id])[-self.median_filter_n:]
            speed_kph = median(recent)
            speed_mps = speed_kph / 3.6
        
        # Apply exponential smoothing
        alpha = 0.3  # Smoothing factor
        smoothed_kph = alpha * speed_kph + (1 - alpha) * prev_kph
        self._smoothed[track_id] = smoothed_kph
        
        return smoothed_kph / 3.6, smoothed_kph, False


# ---------------------------------------------------------------------------
# Violation Tracker for Speed Violation Detection
# ---------------------------------------------------------------------------

class ViolationTracker:
    """
    Tracks speed violations with persistence, cooldown, and capture modes.
    
    CAPTURE-FIRST DESIGN:
    - Events are NOT added to self._events until image is successfully saved.
    - For peak_speed mode, we track per-track state and capture when the track
      drops below threshold OR at end-of-video.
    - Event IDs are assigned only at capture time, not when threshold is crossed.
    - The peak frame is stored per-track to ensure we always have the correct frame.
    
    Per-track state:
    - above_threshold: bool - currently above speed threshold
    - above_since_time: when the track first exceeded the threshold  
    - last_capture_time: cooldown tracking (when last violation was captured)
    - peak_speed, peak_frame_idx, peak_bbox_xyxy, peak_timestamp_s: data for peak capture
    - peak_frame: the actual frame data at peak speed (stored per-track)
    """
    
    def __init__(self, threshold_kmh: float = 110.0, min_seconds: float = 0.3,
                 cooldown_seconds: float = 2.0, capture_mode: str = "peak_speed"):
        self.threshold_kmh = threshold_kmh
        self.min_seconds = min_seconds
        self.cooldown_seconds = cooldown_seconds
        self.capture_mode = capture_mode  # "first_crossing" or "peak_speed"
        
        # Per-track violation state
        self._track_state: Dict[int, dict] = {}
        # List of captured violation events (only added AFTER image is saved)
        self._events: List[dict] = []
        # Event ID is assigned only when event is finalized with image
        self._next_event_id = 1
    
    def _get_state(self, track_id: int) -> dict:
        """Get or create state for a track."""
        if track_id not in self._track_state:
            self._track_state[track_id] = {
                "above_threshold": False,
                "above_since_time": None,
                "last_capture_time": None,  # for cooldown
                "peak_speed": 0.0,
                "peak_frame_idx": None,
                "peak_bbox_xyxy": None,
                "peak_timestamp_s": None,
                "peak_frame": None,  # Store actual frame data for peak
                "qualifies_for_capture": False,  # met min_seconds requirement
            }
        return self._track_state[track_id]
    
    def update(self, track_id: int, speed_kmh: float, t_s: float, frame_idx: int,
               bbox: np.ndarray, class_name: str, frame: np.ndarray = None) -> Optional[dict]:
        """
        Update violation state for a track.
        
        Args:
            track_id: ID of the track
            speed_kmh: current speed
            t_s: current timestamp in seconds
            frame_idx: current frame index
            bbox: bounding box [x1, y1, x2, y2]
            class_name: class name of the object
            frame: the current video frame (REQUIRED for peak_speed mode)
        
        Returns:
            For "first_crossing" mode: returns capture_info dict when ready to capture.
            For "peak_speed" mode: returns capture_info dict when track drops below threshold.
            Returns None if no capture should happen now.
            
        The returned dict contains all info needed to capture, including the frame.
        Caller must save image, then call commit_event() to finalize.
        """
        state = self._get_state(track_id)
        is_above = speed_kmh >= self.threshold_kmh
        
        if is_above:
            # Track is above threshold
            if not state["above_threshold"]:
                # Just crossed above threshold - initialize peak tracking
                state["above_threshold"] = True
                state["above_since_time"] = t_s
                state["peak_speed"] = speed_kmh
                state["peak_frame_idx"] = frame_idx
                state["peak_bbox_xyxy"] = bbox.copy()
                state["peak_timestamp_s"] = t_s
                state["peak_frame"] = frame.copy() if frame is not None else None
                state["qualifies_for_capture"] = False
            else:
                # Continue tracking - update peak if higher speed
                if speed_kmh > state["peak_speed"]:
                    state["peak_speed"] = speed_kmh
                    state["peak_frame_idx"] = frame_idx
                    state["peak_bbox_xyxy"] = bbox.copy()
                    state["peak_timestamp_s"] = t_s
                    state["peak_frame"] = frame.copy() if frame is not None else None
            
            # Check if qualifies for capture (min_seconds met)
            time_above = t_s - state["above_since_time"]
            if time_above >= self.min_seconds:
                state["qualifies_for_capture"] = True
            
            # For first_crossing mode, capture immediately when qualified
            if self.capture_mode == "first_crossing" and state["qualifies_for_capture"]:
                # Check cooldown
                cooldown_ok = (state["last_capture_time"] is None or 
                              (t_s - state["last_capture_time"]) >= self.cooldown_seconds)
                if cooldown_ok:
                    # Return capture info with current frame
                    capture_info = {
                        "track_id": track_id,
                        "class_name": class_name,
                        "speed_kmh": speed_kmh,
                        "frame_idx": frame_idx,
                        "timestamp_s": t_s,
                        "bbox_xyxy": bbox.copy(),
                        "frame": frame.copy() if frame is not None else None,
                    }
                    # Mark as captured to prevent re-capture
                    state["qualifies_for_capture"] = False
                    state["last_capture_time"] = t_s
                    return capture_info
            
            return None
        
        else:
            # Track dropped below threshold
            capture_info = None
            
            if state["above_threshold"] and state["qualifies_for_capture"]:
                # Was above threshold and qualifies - check if should capture
                if self.capture_mode == "peak_speed":
                    # Check cooldown
                    cooldown_ok = (state["last_capture_time"] is None or 
                                  (t_s - state["last_capture_time"]) >= self.cooldown_seconds)
                    if cooldown_ok:
                        # Return capture info for peak speed WITH stored frame
                        capture_info = {
                            "track_id": track_id,
                            "class_name": class_name,
                            "speed_kmh": state["peak_speed"],
                            "frame_idx": state["peak_frame_idx"],
                            "timestamp_s": state["peak_timestamp_s"],
                            "bbox_xyxy": state["peak_bbox_xyxy"].copy(),
                            "frame": state["peak_frame"],  # Use stored peak frame
                        }
                        state["last_capture_time"] = state["peak_timestamp_s"]
            
            # Reset state and free memory
            state["above_threshold"] = False
            state["above_since_time"] = None
            state["peak_speed"] = 0.0
            state["peak_frame_idx"] = None
            state["peak_bbox_xyxy"] = None
            state["peak_timestamp_s"] = None
            state["peak_frame"] = None  # Free frame memory
            state["qualifies_for_capture"] = False
            
            return capture_info
    
    def get_pending_captures(self, class_names: Dict[int, str] = None) -> List[dict]:
        """
        Get all tracks that are still above threshold and qualify for capture.
        Call this at end-of-video to capture remaining violations.
        
        Returns:
            List of capture_info dicts for all pending violations (including frames).
        """
        pending = []
        for track_id, state in self._track_state.items():
            if state["above_threshold"] and state["qualifies_for_capture"]:
                # Check cooldown
                t_s = state["peak_timestamp_s"] or 0
                cooldown_ok = (state["last_capture_time"] is None or 
                              (t_s - state["last_capture_time"]) >= self.cooldown_seconds)
                if cooldown_ok:
                    class_name = class_names.get(track_id, "unknown") if class_names else "unknown"
                    capture_info = {
                        "track_id": track_id,
                        "class_name": class_name,
                        "speed_kmh": state["peak_speed"],
                        "frame_idx": state["peak_frame_idx"],
                        "timestamp_s": state["peak_timestamp_s"],
                        "bbox_xyxy": state["peak_bbox_xyxy"].copy() if state["peak_bbox_xyxy"] is not None else [0,0,0,0],
                        "frame": state["peak_frame"],  # Include stored peak frame
                    }
                    pending.append(capture_info)
        return pending
    
    def commit_event(self, capture_info: dict, image_full: Optional[str], image_crop: Optional[str]) -> dict:
        """
        Commit a capture as a finalized event AFTER images have been saved.
        
        Args:
            capture_info: dict returned by update() or get_pending_captures()
            image_full: path to full frame image (or None if not saved)
            image_crop: path to cropped image (or None if not saved)
        
        Returns:
            The finalized event dict with event_id assigned.
        """
        event = {
            "event_id": self._next_event_id,
            "track_id": capture_info["track_id"],
            "class_name": capture_info["class_name"],
            "speed_kmh": round(capture_info["speed_kmh"], 2),
            "frame_idx": capture_info["frame_idx"],
            "timestamp_s": round(capture_info["timestamp_s"], 3),
            "bbox_xyxy": [round(float(x), 1) for x in capture_info["bbox_xyxy"]],
            "image_full": image_full,
            "image_crop": image_crop,
        }
        self._events.append(event)
        self._next_event_id += 1
        print(f"[VIOLATION] Committed event #{event['event_id']}: track={event['track_id']}, "
              f"speed={event['speed_kmh']}km/h, frame={event['frame_idx']}, "
              f"full={image_full}, crop={image_crop}")
        return event
    
    def get_events(self) -> List[dict]:
        """Get all committed violation events."""
        return self._events
    
    def is_currently_violating(self, track_id: int) -> bool:
        """Check if a track is currently above threshold (for overlay purposes)."""
        state = self._track_state.get(track_id)
        if state is None:
            return False
        return state["above_threshold"]


# ---------------------------------------------------------------------------
# Coordinate Transform Helpers
# ---------------------------------------------------------------------------

def pixel_to_world(u: float, v: float, H: np.ndarray) -> Tuple[float, float]:
    """
    Transform pixel coordinates to world coordinates using homography.
    
    Args:
        u, v: pixel coordinates
        H: 3x3 homography matrix (pixel -> world)
    
    Returns:
        (X_m, Y_m) world coordinates in meters
    """
    # perspectiveTransform expects shape (N, 1, 2)
    pts = np.array([[[u, v]]], dtype=np.float32)
    world_pts = cv2.perspectiveTransform(pts, H)
    return float(world_pts[0, 0, 0]), float(world_pts[0, 0, 1])


def get_contact_point(bbox: np.ndarray) -> Tuple[float, float]:
    """
    Get road contact point (bottom-center) from bounding box.
    
    Args:
        bbox: [x1, y1, x2, y2]
    
    Returns:
        (u, v) pixel coordinates of contact point
    """
    x1, y1, x2, y2 = bbox
    u = (x1 + x2) / 2.0  # center x
    v = y2  # bottom y
    return u, v


def get_anchor_point(bbox: np.ndarray, anchor: str = "bottom_center") -> Tuple[float, float]:
    """
    Get anchor point from bounding box based on anchor type.
    
    Args:
        bbox: [x1, y1, x2, y2]
        anchor: "bottom_center" or "center"
    
    Returns:
        (u, v) pixel coordinates of anchor point
    """
    x1, y1, x2, y2 = bbox
    u = (x1 + x2) / 2.0  # center x
    if anchor == "center":
        v = (y1 + y2) / 2.0  # center y
    else:  # bottom_center (default)
        v = y2  # bottom y
    return u, v


# Curated palette of 24 visually distinct, professional BGR colors
# designed for high contrast on road/asphalt backgrounds
_TRACK_PALETTE_BGR = [
    (230, 159,  56),  # cornflower blue
    ( 80, 200,  50),  # emerald green
    ( 50,  90, 235),  # vermilion red
    ( 50, 210, 245),  # amber/gold
    (200,  80, 180),  # orchid purple
    (  0, 190, 200),  # dark cyan/teal
    (120,  60, 220),  # crimson
    (210, 200,  60),  # sky blue
    ( 40, 170, 100),  # sea green
    (100, 100, 255),  # salmon
    (200, 160,  40),  # steel blue
    ( 60, 220, 180),  # lime
    (180,  50, 130),  # plum
    ( 80, 255, 255),  # yellow
    (255, 144,  30),  # deep sky blue
    ( 30, 105, 210),  # chocolate
    (150, 210,  80),  # light green
    (100,  50, 200),  # dark red
    (255, 200, 100),  # light blue
    ( 50, 180, 255),  # orange
    (200, 100, 200),  # medium purple
    (  0, 215, 175),  # spring green
    (220, 120, 100),  # muted blue
    (130, 230, 200),  # pale green
]


def color_for_id(track_id: int) -> Tuple[int, int, int]:
    """
    Return a deterministic BGR color for a track ID from a curated palette.
    Each vehicle gets a distinct, professional-looking color that stays
    consistent across frames.
    
    Args:
        track_id: The track identifier
    
    Returns:
        (B, G, R) color tuple with values 0-255
    """
    # Use a hash to scatter IDs across the palette so adjacent IDs
    # don't get similar colours
    idx = abs(hash(track_id * 7919)) % len(_TRACK_PALETTE_BGR)
    return _TRACK_PALETTE_BGR[idx]


# ---------------------------------------------------------------------------
# Detection Filtering Functions
# ---------------------------------------------------------------------------

def filter_detections(detections: sv.Detections, settings: dict,
                      frame_width: int, frame_height: int,
                      allowed_class_ids: Optional[set] = None) -> sv.Detections:
    """
    Apply conservative filtering to detections before tracking.
    
    Filters:
    - Class filter (vehicle-like classes)
    - Minimum box area
    - Aspect ratio (optional)
    - ROI (optional)
    - Max detections per frame
    """
    if len(detections) == 0:
        return detections
    
    keep_mask = np.ones(len(detections), dtype=bool)
    
    # 1. Class filter
    if allowed_class_ids is not None and detections.class_id is not None:
        class_mask = np.isin(detections.class_id, list(allowed_class_ids))
        keep_mask &= class_mask
    
    # 2. Minimum box area filter
    min_area = settings.get("min_box_area", 500)
    if min_area > 0:
        areas = (detections.xyxy[:, 2] - detections.xyxy[:, 0]) * \
                (detections.xyxy[:, 3] - detections.xyxy[:, 1])
        keep_mask &= (areas >= min_area)
    
    # 3. Aspect ratio filter (optional)
    if settings.get("aspect_ratio_filter", False):
        ar_min = settings.get("aspect_ratio_min", 0.3)
        ar_max = settings.get("aspect_ratio_max", 4.0)
        widths = detections.xyxy[:, 2] - detections.xyxy[:, 0]
        heights = detections.xyxy[:, 3] - detections.xyxy[:, 1]
        heights = np.maximum(heights, 1)  # avoid division by zero
        aspect_ratios = widths / heights
        keep_mask &= (aspect_ratios >= ar_min) & (aspect_ratios <= ar_max)
    
    # 4. ROI filter (optional)
    if settings.get("roi_enabled", False):
        roi_x1 = settings.get("roi_x1", 0)
        roi_y1 = settings.get("roi_y1", 0)
        roi_x2 = settings.get("roi_x2", frame_width)
        roi_y2 = settings.get("roi_y2", frame_height)
        
        # Check if bbox center is within ROI
        cx = (detections.xyxy[:, 0] + detections.xyxy[:, 2]) / 2
        cy = (detections.xyxy[:, 1] + detections.xyxy[:, 3]) / 2
        in_roi = (cx >= roi_x1) & (cx <= roi_x2) & (cy >= roi_y1) & (cy <= roi_y2)
        keep_mask &= in_roi
    
    # Apply mask
    detections = detections[keep_mask]
    
    # 5. Max detections per frame (keep highest confidence)
    max_det = settings.get("max_det_per_frame", 50)
    if len(detections) > max_det and detections.confidence is not None:
        indices = np.argsort(detections.confidence)[::-1][:max_det]
        detections = detections[indices]
    
    return detections


# ---------------------------------------------------------------------------
# Violation Snapshot Helper
# ---------------------------------------------------------------------------

def save_violation_snapshot(capture_info: dict, event_id: int, frame: np.ndarray, 
                            violations_dir: str, save_full: bool, save_crop: bool, 
                            crop_padding: int, frame_height: int, frame_width: int
                            ) -> Tuple[Optional[str], Optional[str]]:
    """
    Save violation snapshot images (full frame and/or cropped bbox).
    
    Args:
        capture_info: dict with track_id, frame_idx, bbox_xyxy, etc.
        event_id: the event ID to use in filenames
        frame: the video frame to save
        violations_dir: directory to save images
        save_full: whether to save full frame
        save_crop: whether to save cropped bbox
        crop_padding: pixels to add around crop
        frame_height, frame_width: frame dimensions
    
    Returns:
        (image_full_path, image_crop_path) - relative paths like "violations/filename.jpg"
        Returns None for either if not saved or failed.
    """
    track_id = capture_info["track_id"]
    frame_idx = capture_info["frame_idx"]
    bbox = capture_info["bbox_xyxy"]
    
    image_full = None
    image_crop = None

    # Unique filename pattern with event_id, track_id, frame_idx
    crop_filename = f"crop_evt{event_id}_trk{track_id}_frm{frame_idx}.jpg"
    full_filename = f"full_evt{event_id}_trk{track_id}_frm{frame_idx}.jpg"

    # Save full frame first (even if crop fails)
    if save_full:
        full_path = os.path.join(violations_dir, full_filename)
        try:
            ok = cv2.imwrite(full_path, frame, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
            if ok:
                image_full = f"violations/{full_filename}"
                print(f"[VIOLATION] Saved full frame: {image_full}")
            else:
                print(f"[ERROR] cv2.imwrite returned False for full frame: {full_path}")
        except Exception as e:
            print(f"[ERROR] Exception saving full frame {full_path}: {e}")

    # Save cropped bbox
    if save_crop:
        try:
            x1, y1, x2, y2 = [int(v) for v in bbox]
            # Add padding and clamp to frame bounds
            x1_pad = max(0, x1 - crop_padding)
            y1_pad = max(0, y1 - crop_padding)
            x2_pad = min(frame_width, x2 + crop_padding)
            y2_pad = min(frame_height, y2 + crop_padding)
            
            # Ensure valid region (at least 1x1)
            x1_pad = min(max(0, x1_pad), frame_width - 1)
            x2_pad = max(x1_pad + 1, min(x2_pad, frame_width))
            y1_pad = min(max(0, y1_pad), frame_height - 1)
            y2_pad = max(y1_pad + 1, min(y2_pad, frame_height))

            crop = frame[y1_pad:y2_pad, x1_pad:x2_pad]
            if crop.size > 0:
                crop_path = os.path.join(violations_dir, crop_filename)
                ok = cv2.imwrite(crop_path, crop, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
                if ok:
                    image_crop = f"violations/{crop_filename}"
                    print(f"[VIOLATION] Saved crop: {image_crop}")
                else:
                    print(f"[ERROR] cv2.imwrite returned False for crop: {crop_path}")
            else:
                print(f"[ERROR] Crop region empty for track={track_id}, frame={frame_idx}, bbox={bbox}")
        except Exception as e:
            print(f"[ERROR] Exception saving crop for track={track_id}: {e}")
    
    return image_full, image_crop


# ---------------------------------------------------------------------------
# Main Analysis Function
# ---------------------------------------------------------------------------

def run_speed_job(job_dir: str, settings: dict = None, progress_cb=None, message_cb=None) -> dict:
    """
    Run speed analysis on a calibrated video.
    
    Args:
        job_dir: Path to job directory containing input.mp4 and calibration.json
        settings: Optional settings dict. If None, uses defaults + env vars.
        progress_cb: Optional callback for progress updates (0.0 to 1.0)
        message_cb: Optional callback for status messages
    
    Returns:
        Summary dict with analysis results
    
    Raises:
        RuntimeError: If required files are missing or analysis fails
    """
    if progress_cb is None:
        progress_cb = lambda p: None
    if message_cb is None:
        message_cb = lambda m: None
    
    # Merge settings with defaults
    effective_settings = get_default_settings()
    if settings:
        effective_settings.update(settings)
    
    # Compute settings hash for cache fingerprinting
    settings_fingerprint = _settings_hash(effective_settings)
    
    # Check for cached results with matching settings
    output_video = os.path.join(job_dir, "output.mp4")
    output_csv = os.path.join(job_dir, "tracks_v1.csv")
    output_json = os.path.join(job_dir, "summary.json")
    
    if os.path.exists(output_video) and os.path.exists(output_csv) and os.path.exists(output_json):
        # Check if cached results match current settings
        try:
            with open(output_json, "r") as f:
                cached_summary = json.load(f)
            cached_fingerprint = cached_summary.get("settings_fingerprint", "")
            if cached_fingerprint == settings_fingerprint:
                message_cb("Using cached results (settings unchanged)")
                return cached_summary
            else:
                message_cb("Settings changed, re-running analysis...")
        except Exception:
            pass
    
    # Load calibration
    cal_path = os.path.join(job_dir, "calibration.json")
    if not os.path.exists(cal_path):
        raise RuntimeError("Calibration not found. Please calibrate first.")
    
    with open(cal_path, "r") as f:
        cal_data = json.load(f)
    
    H = np.array(cal_data["H_pixel_to_world"], dtype=np.float32)
    if H.shape != (3, 3):
        raise RuntimeError("Invalid homography matrix in calibration.json")
    
    # Open input video
    input_video = os.path.join(job_dir, "input.mp4")
    if not os.path.exists(input_video):
        raise RuntimeError("Input video not found")
    
    message_cb("Opening video...")
    cap = cv2.VideoCapture(input_video)
    if not cap.isOpened():
        raise RuntimeError("Failed to open input video")
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Handle resize option
    resize_width = effective_settings.get("resize_width", 0)
    inference_width, inference_height = width, height
    scale_factor = 1.0
    if resize_width > 0 and resize_width < width:
        scale_factor = width / resize_width
        inference_width = resize_width
        inference_height = int(height / scale_factor)
        message_cb(f"Resizing for inference: {inference_width}x{inference_height}")
    
    # Get frame_skip setting
    frame_skip = effective_settings.get("frame_skip", 1)
    
    # Initialize output video writer (always at original resolution)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    
    # Adjust output fps for frame_skip
    output_fps = fps / frame_skip
    
    writer = cv2.VideoWriter(output_video, fourcc, output_fps, (width, height))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError("Failed to create output video writer")
    
    # Load configuration from settings
    max_kph = effective_settings.get("max_kph", 160.0)
    window_s = effective_settings.get("smoothing_window_s", 0.7)
    min_dt_s = effective_settings.get("min_dt_s", 0.1)
    conf_threshold = effective_settings.get("min_conf", 0.25)
    outlier_reject = effective_settings.get("outlier_reject_kph_per_s", 50.0)
    median_n = effective_settings.get("median_filter_n", 5)
    jump_reject = effective_settings.get("jump_reject_m", 15.0)
    
    # Parse class filter if provided
    allowed_class_ids = None
    class_filter = effective_settings.get("class_filter", "")
    if class_filter.strip():
        try:
            allowed_class_ids = set(int(x.strip()) for x in class_filter.split(",") if x.strip())
        except ValueError:
            allowed_class_ids = None
    
    # Initialize YOLO model
    message_cb("Loading YOLO model...")
    model_path = os.path.join(os.path.dirname(__file__), "..", "..", "..", "yolo11n.pt")
    if not os.path.exists(model_path):
        # Try current working directory
        model_path = "yolo11n.pt"
    
    model = YOLO(model_path)
    
    # Initialize ByteTrack tracker ONCE for the entire video
    # (not per-frame, as that would reset IDs)
    tracker = sv.ByteTrack(
        track_activation_threshold=effective_settings.get("track_thresh", 0.25),
        lost_track_buffer=effective_settings.get("track_buffer", 30),
        minimum_matching_threshold=effective_settings.get("match_thresh", 0.8),
        frame_rate=int(fps / frame_skip)  # Adjust for frame_skip
    )
    
    # Initialize annotators with per-track colors from our curated palette
    _sv_palette = sv.ColorPalette(
        [sv.Color(r=c[2], g=c[1], b=c[0]) for c in _TRACK_PALETTE_BGR]
    )
    box_annotator = sv.BoxAnnotator(
        thickness=2,
        color=_sv_palette,
        color_lookup=sv.ColorLookup.TRACK,
    )
    label_annotator = sv.LabelAnnotator(
        text_position=sv.Position.TOP_LEFT,
        color=_sv_palette,
        text_color=sv.Color.WHITE,
        text_scale=0.5,
        text_thickness=1,
        text_padding=8,
        color_lookup=sv.ColorLookup.TRACK,
        border_radius=4,
    )
    
    # Initialize speed calculator with new parameters
    speed_calc = SpeedCalculator(
        window_s=window_s,
        min_dt_s=min_dt_s,
        max_kph=max_kph,
        outlier_reject_kph_per_s=outlier_reject,
        median_filter_n=median_n,
        jump_reject_m=jump_reject
    )
    track_stats = SpeedTrackStats()
    
    # ---------------------------------------------------------------------------
    # Trail visualization setup (motion history)
    # ---------------------------------------------------------------------------
    enable_trails = effective_settings.get("enable_trails", False)
    trail_seconds = effective_settings.get("trail_seconds", 2.0)
    trail_max_points = effective_settings.get("trail_max_points", 60)
    trail_stride = max(1, effective_settings.get("trail_stride", 1))
    trail_thickness = effective_settings.get("trail_thickness", 2)
    trail_alpha = effective_settings.get("trail_alpha", 0.85)
    trail_fade = effective_settings.get("trail_fade", True)
    trail_anchor = effective_settings.get("trail_anchor", "bottom_center")
    
    # Compute max history frames based on trail_seconds and frame timing
    # Account for frame_skip: we only process every Nth frame
    effective_fps = fps / frame_skip
    max_history_frames = min(trail_max_points, int(trail_seconds * effective_fps))
    max_history_frames = max(2, max_history_frames)  # at least 2 points for a line
    
    # Trail history: dict[track_id, deque[(u, v)]] storing pixel coords
    trail_histories: Dict[int, deque] = defaultdict(lambda: deque(maxlen=max_history_frames))
    # Track last seen frame for pruning old tracks
    trail_last_seen: Dict[int, int] = {}
    
    # ---------------------------------------------------------------------------
    # Violation detection setup
    # ---------------------------------------------------------------------------
    violation_enabled = effective_settings.get("violation_enabled", False)
    violation_speed_kmh = effective_settings.get("violation_speed_kmh", 110.0)
    violation_min_seconds = effective_settings.get("violation_min_seconds", 0.3)
    violation_cooldown_seconds = effective_settings.get("violation_cooldown_seconds", 2.0)
    violation_capture_mode = effective_settings.get("violation_capture_mode", "peak_speed")
    violation_save_full = effective_settings.get("violation_save_full_frame", True)
    violation_save_crop = effective_settings.get("violation_save_crop", True)
    violation_crop_padding = effective_settings.get("violation_crop_padding_px", 20)
    
    # Initialize violation tracker (only if enabled)
    violation_tracker = None
    violations_dir = None
    # Track class names for finalize (needed at end of video)
    track_class_names: Dict[int, str] = {}
    # Note: Frames are now stored per-track inside ViolationTracker (no external storage needed)
    
    if violation_enabled:
        violation_tracker = ViolationTracker(
            threshold_kmh=violation_speed_kmh,
            min_seconds=violation_min_seconds,
            cooldown_seconds=violation_cooldown_seconds,
            capture_mode=violation_capture_mode
        )
        violations_dir = os.path.join(job_dir, "violations")
        os.makedirs(violations_dir, exist_ok=True)
        message_cb(f"Violation detection enabled (threshold: {violation_speed_kmh} km/h)")
    
    # Open CSV for writing
    message_cb("Processing frames...")
    csv_file = open(output_csv, "w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow([
        "frame", "time_s", "track_id", "class_name", "conf",
        "x1", "y1", "x2", "y2", "u", "v",
        "world_x_m", "world_y_m", "speed_mps", "speed_kph", "rejected"
    ])
    
    frame_idx = 0
    processed_frames = 0
    overall_max_kph = 0.0
    total_rejected = 0
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_idx += 1
            
            # Frame skip: only process every Nth frame
            if frame_skip > 1 and (frame_idx - 1) % frame_skip != 0:
                continue
            
            processed_frames += 1
            t_s = frame_idx / fps  # Time in original video timeline
            
            # Resize for inference if configured
            inference_frame = frame
            if scale_factor > 1.0:
                inference_frame = cv2.resize(frame, (inference_width, inference_height))
            
            # Run YOLO detection
            results = model(inference_frame, conf=conf_threshold, verbose=False)[0]
            detections = sv.Detections.from_ultralytics(results)
            
            # Scale bboxes back to original resolution if resized
            if scale_factor > 1.0 and len(detections) > 0:
                detections.xyxy = detections.xyxy * scale_factor
            
            # Apply conservative filtering BEFORE tracker
            detections = filter_detections(
                detections, effective_settings, width, height, allowed_class_ids
            )
            
            # Update tracker with filtered detections
            detections = tracker.update_with_detections(detections)
            
            # Process each detection
            labels = []
            current_frame_track_ids = set()  # Track IDs seen in this frame
            
            for i in range(len(detections)):
                bbox = detections.xyxy[i]
                track_id = int(detections.tracker_id[i]) if detections.tracker_id is not None else -1
                class_id = int(detections.class_id[i]) if detections.class_id is not None else -1
                conf = float(detections.confidence[i]) if detections.confidence is not None else 0.0
                
                # Get class name
                class_name = model.names.get(class_id, str(class_id)) if hasattr(model, 'names') else str(class_id)
                
                # Get contact point (bottom-center)
                u, v = get_contact_point(bbox)
                
                # Transform to world coordinates
                try:
                    world_x, world_y = pixel_to_world(u, v, H)
                except Exception:
                    world_x, world_y = 0.0, 0.0
                
                # Calculate speed with outlier rejection
                speed_mps, speed_kph, was_rejected = 0.0, 0.0, False
                if track_id >= 0:
                    speed_mps, speed_kph, was_rejected = speed_calc.update(track_id, t_s, world_x, world_y)
                    track_stats.update(track_id, frame_idx, class_name, speed_kph, rejected=was_rejected)
                    
                    if was_rejected:
                        total_rejected += 1
                    elif speed_kph > overall_max_kph:
                        overall_max_kph = speed_kph
                    
                    # Update trail history if trails are enabled
                    if enable_trails:
                        current_frame_track_ids.add(track_id)
                        # Compute anchor point for trail
                        trail_u, trail_v = get_anchor_point(bbox, trail_anchor)
                        trail_u = max(0, min(width - 1, trail_u))
                        trail_v = max(0, min(height - 1, trail_v))
                        # Append to history
                        trail_histories[track_id].append((trail_u, trail_v))
                        trail_last_seen[track_id] = processed_frames
                    
                    # Track class name for violation finalize
                    if violation_enabled:
                        track_class_names[track_id] = class_name
                
                # ---------------------------------------------------------------------------
                # Violation detection (if enabled) - CAPTURE-FIRST design
                # Frame is stored per-track inside ViolationTracker for peak_speed mode
                # ---------------------------------------------------------------------------
                is_violating = False
                if violation_enabled and violation_tracker is not None and track_id >= 0 and not was_rejected:
                    # Check if currently violating (for overlay)
                    is_violating = violation_tracker.is_currently_violating(track_id)
                    
                    # Update violation tracker - pass frame for storage
                    # Returns capture_info (with frame) if ready to capture
                    capture_info = violation_tracker.update(
                        track_id, speed_kph, t_s, frame_idx, bbox, class_name, frame
                    )
                    
                    # If capture_info returned, save image THEN commit event
                    if capture_info is not None:
                        # Get the next event_id (before committing)
                        next_event_id = violation_tracker._next_event_id
                        # Use frame from capture_info (stored at peak moment)
                        frame_to_save = capture_info.get("frame")
                        if frame_to_save is None:
                            print(f"[WARNING] No frame in capture_info for track {track_id}, using current frame")
                            frame_to_save = frame
                        # Save snapshot
                        image_full, image_crop = save_violation_snapshot(
                            capture_info, next_event_id, frame_to_save.copy(), violations_dir,
                            violation_save_full, violation_save_crop, violation_crop_padding, height, width
                        )
                        # Commit event only after images saved
                        violation_tracker.commit_event(capture_info, image_full, image_crop)
                
                # Create label (ASCII-only) - add VIOLATION marker if above threshold
                if violation_enabled and is_violating and speed_kph >= violation_speed_kmh:
                    label = f"ID:{track_id} {speed_kph:.1f} km/h VIOLATION"
                elif speed_kph > 0.5:
                    label = f"ID:{track_id} {speed_kph:.1f} km/h"
                else:
                    label = f"ID:{track_id}"
                labels.append(label)
                
                # Write to CSV
                csv_writer.writerow([
                    frame_idx,
                    f"{t_s:.3f}",
                    track_id,
                    class_name,
                    f"{conf:.3f}",
                    f"{bbox[0]:.1f}", f"{bbox[1]:.1f}", f"{bbox[2]:.1f}", f"{bbox[3]:.1f}",
                    f"{u:.1f}", f"{v:.1f}",
                    f"{world_x:.3f}", f"{world_y:.3f}",
                    f"{speed_mps:.3f}", f"{speed_kph:.2f}",
                    "1" if was_rejected else "0"
                ])
            
            # Prune old trails (tracks not seen for a while)
            if enable_trails:
                stale_tracks = [
                    tid for tid, last_frame in trail_last_seen.items()
                    if processed_frames - last_frame > max_history_frames
                ]
                for tid in stale_tracks:
                    if tid in trail_histories:
                        del trail_histories[tid]
                    if tid in trail_last_seen:
                        del trail_last_seen[tid]
            
            # ---------------------------------------------------------------------------
            # Draw trails BEFORE bbox/label annotation (so labels stay on top)
            # Uses anti-aliased lines with gradient opacity and tapering thickness
            # ---------------------------------------------------------------------------
            if enable_trails and len(current_frame_track_ids) > 0:
                # Always use an overlay for proper alpha blending
                overlay = frame.copy()
                
                for track_id in current_frame_track_ids:
                    history = trail_histories.get(track_id)
                    if history is None or len(history) < 2:
                        continue
                    
                    # Get points with stride
                    pts_list = list(history)[::trail_stride]
                    if len(pts_list) < 2:
                        continue
                    
                    # Get the track's unique colour
                    color = color_for_id(track_id)
                    n_segs = len(pts_list) - 1
                    
                    for seg_idx in range(n_segs):
                        pt1 = (int(pts_list[seg_idx][0]), int(pts_list[seg_idx][1]))
                        pt2 = (int(pts_list[seg_idx + 1][0]), int(pts_list[seg_idx + 1][1]))
                        
                        # age_factor: 0 = oldest segment, 1 = newest
                        age_factor = (seg_idx + 1) / n_segs
                        
                        if trail_fade:
                            # Fade colour: older → darker
                            intensity = 0.25 + 0.75 * age_factor  # range [0.25, 1.0]
                            seg_color = (
                                int(color[0] * intensity),
                                int(color[1] * intensity),
                                int(color[2] * intensity)
                            )
                        else:
                            seg_color = color
                        
                        # Taper thickness: oldest = 1px, newest = trail_thickness
                        seg_thick = max(1, int(1 + (trail_thickness - 1) * age_factor))
                        
                        # Draw anti-aliased line
                        cv2.line(overlay, pt1, pt2, seg_color, seg_thick, cv2.LINE_AA)
                
                # Blend the trail overlay onto the frame
                cv2.addWeighted(overlay, trail_alpha, frame, 1 - trail_alpha, 0, frame)
            
            # Annotate frame (bboxes and labels ON TOP of trails)
            frame = box_annotator.annotate(frame, detections)
            frame = label_annotator.annotate(frame, detections, labels=labels)
            
            # Write frame
            writer.write(frame)
            
            # Update progress
            if total_frames > 0:
                progress_cb(min(0.99, frame_idx / total_frames))
            
    finally:
        csv_file.close()
        cap.release()
        writer.release()
    
    # Re-encode video for browser compatibility (Chrome, Firefox, Edge)
    message_cb("Re-encoding video for browser compatibility...")
    if reencode_video_for_browser(output_video):
        message_cb("Video re-encoded to H.264 for browser compatibility")
    else:
        message_cb("Note: ffmpeg not available, video may not play in Chrome/Firefox")
    
    # ---------------------------------------------------------------------------
    # Finalize violations (capture any pending peak_speed events at end-of-video)
    # ---------------------------------------------------------------------------
    violation_summary = {"enabled": violation_enabled, "count": 0, "events": []}

    images_saved = 0
    images_missing = 0

    if violation_enabled and violation_tracker is not None:
        # Get all pending captures (tracks still above threshold at end of video)
        # Each capture_info includes the stored peak frame
        pending_captures = violation_tracker.get_pending_captures(track_class_names)
        print(f"[VIOLATION] End-of-video: {len(pending_captures)} pending captures to finalize")

        # Process each pending capture - save image THEN commit
        for capture_info in pending_captures:
            # Get the next event_id (before committing)
            next_event_id = violation_tracker._next_event_id
            
            # Use frame from capture_info (stored at peak moment)
            frame_to_save = capture_info.get("frame")
            if frame_to_save is not None:
                image_full, image_crop = save_violation_snapshot(
                    capture_info, next_event_id, frame_to_save.copy(), violations_dir,
                    violation_save_full, violation_save_crop, violation_crop_padding, height, width
                )
                # Commit event with image paths
                violation_tracker.commit_event(capture_info, image_full, image_crop)
            else:
                # Frame not available - still commit event but without images
                print(f"[WARNING] No frame stored for track {capture_info['track_id']}, "
                      f"committing event without images")
                violation_tracker.commit_event(capture_info, None, None)

        # Get all committed events
        all_events = violation_tracker.get_events()
        violation_summary["count"] = len(all_events)
        violation_summary["events"] = all_events
        violation_summary["threshold_kmh"] = violation_speed_kmh
        violation_summary["capture_mode"] = violation_capture_mode

        # Count images saved/missing (at least one image must exist)
        for evt in all_events:
            has_image = evt.get("image_crop") or evt.get("image_full")
            if has_image:
                images_saved += 1
            else:
                images_missing += 1

        violation_summary["images_saved"] = images_saved
        violation_summary["images_missing"] = images_missing
        
        # Debug assertion - verify image counts
        violation_summary["violations_total"] = len(all_events)
        violation_summary["violations_with_images"] = images_saved
        violation_summary["violations_missing_images"] = images_missing
        
        if images_missing > 0:
            print(f"[WARNING] {images_missing}/{len(all_events)} violations have no images!")

        # Write violations.json
        if violations_dir:
            violations_json_path = os.path.join(violations_dir, "violations.json")
            with open(violations_json_path, "w") as f:
                json.dump(all_events, f, indent=2)

            # Write violations.csv
            violations_csv_path = os.path.join(violations_dir, "violations.csv")
            with open(violations_csv_path, "w", newline="") as f:
                csv_w = csv.writer(f)
                csv_w.writerow(["event_id", "track_id", "class_name", "speed_kmh", 
                               "frame_idx", "timestamp_s", "bbox_x1", "bbox_y1", 
                               "bbox_x2", "bbox_y2", "image_full", "image_crop"])
                for evt in all_events:
                    bbox = evt["bbox_xyxy"]
                    csv_w.writerow([
                        evt["event_id"], evt["track_id"], evt["class_name"],
                        evt["speed_kmh"], evt["frame_idx"], evt["timestamp_s"],
                        bbox[0], bbox[1], bbox[2], bbox[3],
                        evt.get("image_full", ""), evt.get("image_crop", "")
                    ])

        message_cb(f"Detected {len(all_events)} speed violations, images saved: {images_saved}, missing: {images_missing}")
    
    # Write summary JSON
    per_track = track_stats.get_summary()
    summary = {
        "video": os.path.basename(input_video),
        "fps": round(fps, 2),
        "output_fps": round(output_fps, 2),
        "frames_total": total_frames,
        "frames_processed": processed_frames,
        "frame_skip": frame_skip,
        "unique_tracks": track_stats.unique_tracks,
        "overall_max_kph": round(overall_max_kph, 2),
        "total_rejected_updates": total_rejected,
        "settings_fingerprint": settings_fingerprint,
        "settings_used": effective_settings,
        "source_page": "speed",
        "analysis_type": "speed_calibrated",
        "created_at": datetime.now().isoformat(),
        "config": {
            "max_kph": max_kph,
            "window_s": window_s,
            "min_dt_s": min_dt_s,
            "conf_threshold": conf_threshold,
            "outlier_reject_kph_per_s": outlier_reject,
            "median_filter_n": median_n,
        },
        "per_track": per_track,
        "calibration": {
            "pixel_points": cal_data.get("pixel_points"),
            "world_points_m": cal_data.get("world_points_m"),
        },
        "violations": violation_summary,
    }
    
    with open(output_json, "w") as f:
        json.dump(summary, f, indent=2)
    
    message_cb("Done.")
    progress_cb(1.0)
    
    return summary
