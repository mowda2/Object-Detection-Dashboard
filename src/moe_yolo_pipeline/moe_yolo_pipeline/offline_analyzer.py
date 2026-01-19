import os, time, csv, json, tempfile, shutil, subprocess
from typing import Dict, Iterable, Optional
from collections import defaultdict, deque
import numpy as np
import cv2
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
        cmd = [
            ffmpeg_path, "-y", "-i", input_path,
            "-c:v", "libx264", "-preset", "fast", "-crf", "23",
            "-pix_fmt", "yuv420p", "-movflags", "+faststart",
            "-an",  # no audio
            temp_output
        ]
        
        result = subprocess.run(cmd, capture_output=True, timeout=300)
        
        if result.returncode == 0 and os.path.exists(temp_output):
            os.replace(temp_output, output_path)
            return True
        else:
            if os.path.exists(temp_output):
                os.remove(temp_output)
            return False
            
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Environment-variable-based configuration for tracking (Phase 1-2 improvements)
# All defaults preserve existing behavior when env vars are not set.
# ---------------------------------------------------------------------------

def _env_float(name: str, default: float) -> float:
    """Read a float from env var, returning default if not set or invalid."""
    val = os.environ.get(name, "")
    if not val:
        return default
    try:
        return float(val)
    except ValueError:
        return default

def _env_int(name: str, default: int) -> int:
    """Read an int from env var, returning default if not set or invalid."""
    val = os.environ.get(name, "")
    if not val:
        return default
    try:
        return int(val)
    except ValueError:
        return default

def _env_bool(name: str, default: bool = False) -> bool:
    """Read a boolean from env var (1/true/yes = True)."""
    val = os.environ.get(name, "").strip().lower()
    if not val:
        return default
    return val in ("1", "true", "yes", "on")


# Tracking filter env vars (defaults preserve current behavior)
MOE_TRACK_MIN_CONF = lambda: _env_float("MOE_TRACK_MIN_CONF", 0.0)        # No extra filtering by default
MOE_TRACK_MIN_BOX_AREA = lambda: _env_int("MOE_TRACK_MIN_BOX_AREA", 0)    # No area filtering by default
MOE_TRACK_CLASSES = lambda: os.environ.get("MOE_TRACK_CLASSES", "")       # Empty = no extra class filter

# ByteTrack tuning env vars (supervision sv.ByteTrack parameters)
# track_activation_threshold: minimum confidence to start a new track
# lost_track_buffer: frames to keep lost tracks before removal
# minimum_matching_threshold: IOU threshold for matching
# frame_rate: used internally by ByteTrack
MOE_TRACK_BUFFER = lambda: _env_int("MOE_TRACK_BUFFER", 30)               # Default 30 (sv.ByteTrack default)
MOE_TRACK_THRESH = lambda: _env_float("MOE_TRACK_THRESH", 0.25)           # track_activation_threshold
MOE_TRACK_MATCH_THRESH = lambda: _env_float("MOE_TRACK_MATCH_THRESH", 0.8) # minimum_matching_threshold

# Trail visualization env vars (disabled by default)
MOE_SHOW_TRAILS = lambda: _env_bool("MOE_SHOW_TRAILS", False)
MOE_TRAIL_LEN = lambda: _env_int("MOE_TRAIL_LEN", 25)

# ---------------------------------------------------------------------------
# PASS 2: Professional outputs env vars (all disabled by default)
# ---------------------------------------------------------------------------
MOE_SHOW_HUD = lambda: _env_bool("MOE_SHOW_HUD", False)        # On-screen HUD overlay
MOE_WRITE_CSV_V2 = lambda: _env_bool("MOE_WRITE_CSV_V2", False)  # Extended CSV with more columns


# ---------------------------------------------------------------------------
# PASS 2: TrackStats collector for per-tracklet statistics
# ---------------------------------------------------------------------------
class TrackStats:
    """
    Lightweight tracklet statistics collector for per-video analysis.
    Collects start_frame, end_frame, frames_seen, avg_conf, avg_area, etc.
    All updates are O(1) per detection.
    """
    
    def __init__(self):
        self._tracks: Dict[int, dict] = {}  # track_id -> stats dict
        self._unique_count = 0
        
    def update(self, track_id: int, frame_idx: int, class_name: str,
               confidence: float, box_area: float):
        """
        Update stats for a track_id. Call once per detection per frame.
        Uses running mean for avg_conf and avg_area (O(1) updates).
        """
        if track_id not in self._tracks:
            # New track
            self._tracks[track_id] = {
                "track_id": track_id,
                "class": class_name,
                "start_frame": frame_idx,
                "end_frame": frame_idx,
                "frames_seen": 1,
                "avg_conf": confidence,
                "avg_area": box_area,
                "max_conf": confidence,
                "_conf_sum": confidence,  # internal for running mean
                "_area_sum": box_area,
            }
            self._unique_count += 1
        else:
            s = self._tracks[track_id]
            s["end_frame"] = frame_idx
            s["frames_seen"] += 1
            s["_conf_sum"] += confidence
            s["_area_sum"] += box_area
            s["avg_conf"] = s["_conf_sum"] / s["frames_seen"]
            s["avg_area"] = s["_area_sum"] / s["frames_seen"]
            if confidence > s["max_conf"]:
                s["max_conf"] = confidence
    
    @property
    def unique_tracks(self) -> int:
        """Total unique track IDs seen so far."""
        return self._unique_count
    
    def active_tracks_in_frame(self, track_ids: Iterable[int]) -> int:
        """Count how many of the given track_ids are known (active this frame)."""
        return sum(1 for tid in track_ids if tid in self._tracks)
    
    def export_summary(self, video_name: str, fps: float, frames_processed: int) -> dict:
        """
        Export summary as dict suitable for JSON serialization.
        """
        tracks_list = []
        for tid, s in sorted(self._tracks.items()):
            tracks_list.append({
                "track_id": s["track_id"],
                "class": s["class"],
                "start_frame": s["start_frame"],
                "end_frame": s["end_frame"],
                "frames_seen": s["frames_seen"],
                "avg_conf": round(s["avg_conf"], 4),
                "avg_area": round(s["avg_area"], 2),
                "max_conf": round(s["max_conf"], 4),
            })
        return {
            "video": video_name,
            "fps": round(fps, 2),
            "frames_processed": frames_processed,
            "unique_tracks": self._unique_count,
            "tracks": tracks_list,
        }


def _write_summary_json_safe(summary: dict, output_path: str, message_cb=None):
    """
    Write summary JSON atomically (temp file + rename). 
    Never crashes pipeline on failure - just logs and continues.
    """
    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        # Write to temp file in same directory (for atomic rename)
        fd, tmp_path = tempfile.mkstemp(suffix=".json", dir=os.path.dirname(output_path))
        try:
            with os.fdopen(fd, "w") as f:
                json.dump(summary, f, indent=2)
            shutil.move(tmp_path, output_path)
        except Exception:
            # Clean up temp file if rename failed
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
            raise
    except Exception as e:
        if message_cb:
            message_cb(f"Warning: Failed to write summary.json: {e}")
        # Don't re-raise - pipeline continues


# ---------------------------------------------------------------------------
# PASS 2: HUD overlay drawing
# ---------------------------------------------------------------------------
def _draw_hud(frame: np.ndarray, frame_idx: int, active_tracks: int,
              unique_tracks: int, proc_fps: float, tracking_enabled: bool = True):
    """
    Draw a small HUD overlay in the top-left corner of the frame.
    ASCII-only, minimal, non-intrusive.
    
    Args:
        frame: BGR image (modified in place)
        frame_idx: current frame number
        active_tracks: number of tracks visible this frame
        unique_tracks: total unique tracks seen so far
        proc_fps: processing FPS (frames per second throughput)
        tracking_enabled: whether tracking is active
    """
    # HUD lines
    lines = [
        f"Frame: {frame_idx}",
        f"Active: {active_tracks}",
        f"Unique: {unique_tracks}",
        f"FPS: {proc_fps:.1f}",
        f"Track: {'ON' if tracking_enabled else 'OFF'}",
    ]
    
    # Drawing params
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    thickness = 1
    color = (0, 255, 0)  # Green
    bg_color = (0, 0, 0)  # Black background
    padding = 5
    line_height = 18
    
    # Calculate background rectangle size
    max_width = 0
    for line in lines:
        (w, h), _ = cv2.getTextSize(line, font, font_scale, thickness)
        max_width = max(max_width, w)
    
    bg_width = max_width + 2 * padding
    bg_height = len(lines) * line_height + 2 * padding
    
    # Draw semi-transparent background
    overlay = frame.copy()
    cv2.rectangle(overlay, (5, 5), (5 + bg_width, 5 + bg_height), bg_color, -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
    
    # Draw text
    y_offset = 5 + padding + 14  # Start after padding + baseline
    for line in lines:
        cv2.putText(frame, line, (5 + padding, y_offset), font, font_scale, color, thickness, cv2.LINE_AA)
        y_offset += line_height


def _center_xyxy(xyxy):
    """Compute center (cx, cy) from xyxy bounding box."""
    x1, y1, x2, y2 = xyxy
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def _box_area(xyxy) -> float:
    """Compute box area from xyxy."""
    x1, y1, x2, y2 = xyxy
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)

def _normalize_names(names: Iterable[str]) -> set:
    alias = {
        "people": "person",
        "pedestrian": "person", "pedestrians": "person",
        "bike": "bicycle", "bikes": "bicycle", "cycle": "bicycle",
        "motorbike": "motorcycle", "motorbikes": "motorcycle",
        "van": "truck"
    }
    out = set()
    for n in names:
        n = (n or "").strip().lower()
        if not n: continue
        out.add(alias.get(n, n))
    return out

def _resolve_allowed_ids(model_names, include_names) -> set:
    """Resolve class names to class IDs based on model.names mapping."""
    if isinstance(model_names, dict):
        inv = {v.lower(): int(k) for k, v in model_names.items()}
    else:
        inv = {str(v).lower(): i for i, v in enumerate(model_names)}
    ids = set()
    for n in include_names:
        if n in inv:
            ids.add(inv[n])
    if not ids:
        coco_guess = {"person":0,"bicycle":1,"car":2,"motorcycle":3,"bus":5,"truck":7}
        for n in include_names:
            if n in coco_guess: ids.add(coco_guess[n])
    return ids


def _filter_detections_pre_track(det: sv.Detections, min_conf: float, min_area: int, 
                                  allowed_class_ids: set = None) -> sv.Detections:
    """
    Apply pre-tracking filters to remove noisy detections.
    This reduces ID switches by not feeding low-quality detections into the tracker.
    
    Args:
        det: sv.Detections object
        min_conf: minimum confidence threshold (0.0 = no filtering)
        min_area: minimum box area in pixels (0 = no filtering)
        allowed_class_ids: set of class IDs to keep (None = keep all)
    
    Returns:
        Filtered sv.Detections
    """
    if len(det) == 0:
        return det
    
    # Start with all True mask
    keep = np.ones(len(det), dtype=bool)
    
    # Confidence filter
    if min_conf > 0.0 and det.confidence is not None:
        keep &= (det.confidence >= min_conf)
    
    # Box area filter
    if min_area > 0:
        areas = np.array([_box_area(xyxy) for xyxy in det.xyxy])
        keep &= (areas >= min_area)
    
    # Class ID filter (if specified via MOE_TRACK_CLASSES env var)
    if allowed_class_ids is not None and len(allowed_class_ids) > 0:
        if det.class_id is not None and len(det.class_id) > 0:
            keep &= np.isin(det.class_id, list(allowed_class_ids))
    
    return det[keep]


def _draw_trails(frame: np.ndarray, trails: Dict[int, deque], color=(0, 255, 255), thickness=2):
    """
    Draw track trails (recent centroid history) as polylines.
    
    Args:
        frame: BGR image to draw on (modified in place)
        trails: dict mapping track_id -> deque of (cx, cy) tuples
        color: BGR color for trails
        thickness: line thickness
    """
    for tid, pts in trails.items():
        if len(pts) < 2:
            continue
        # Convert deque to numpy array for cv2.polylines
        pts_arr = np.array(list(pts), dtype=np.int32).reshape((-1, 1, 2))
        cv2.polylines(frame, [pts_arr], isClosed=False, color=color, thickness=thickness)


def _format_label(class_name: str, track_id: int, confidence: float, speed_kmh: float = None) -> str:
    """
    Format a clean, ASCII-only detection label.
    
    Format: "class #ID conf" or "class #ID conf speed" if speed provided
    Example: "person #62 0.83" or "car #15 0.91 45.2km/h"
    """
    # Ensure ASCII-only class name (replace any non-ASCII)
    clean_name = class_name.encode('ascii', 'replace').decode('ascii')
    
    if speed_kmh is not None and speed_kmh > 0.1:
        return f"{clean_name} #{track_id} {confidence:.2f} {speed_kmh:.1f}km/h"
    else:
        return f"{clean_name} #{track_id} {confidence:.2f}"

def run_offline_speed_job(job_rec: Dict):
    """
    Inputs in job_rec:
      src, out_video, out_csv, [out_json], [poster_path]
    Options:
      model_path, conf, meters_per_pixel, device ('cuda'|'cpu'), include (names),
      progress_cb, message_cb
    """
    progress_cb = job_rec.get("progress_cb", lambda p: None)
    message_cb = job_rec.get("message_cb", lambda m: None)

    try:
        src = job_rec["src"]
        out_video = job_rec["out_video"]
        out_csv = job_rec["out_csv"]
        out_json = job_rec.get("out_json")
        poster_path = job_rec.get("poster_path")

        model_path = job_rec.get("model_path", "yolo11n.pt")
        conf = float(job_rec.get("conf", 0.25))
        meters_per_pixel = float(job_rec.get("meters_per_pixel", 0.05))
        device = job_rec.get("device", "cuda")

        include = job_rec.get("include", "")
        if isinstance(include, str):
            include_names = [s.strip() for s in include.split(",") if s.strip()]
        else:
            include_names = list(include or [])
        if not include_names:
            include_names = ["car", "motorcycle", "bus", "truck"]
        include_names = _normalize_names(include_names)

        message_cb("Opening video…")
        cap = cv2.VideoCapture(src)
        if not cap.isOpened():
            raise RuntimeError("Failed to open input video")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Save poster from first frame (if asked)
        if poster_path:
            pos = int(cap.get(cv2.CAP_PROP_POS_FRAMES) or 0)
            ok, fr = cap.read()
            if ok:
                cv2.imwrite(poster_path, fr, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
            cap.set(cv2.CAP_PROP_POS_FRAMES, pos)

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(out_video, fourcc, fps, (width, height))
        if not writer.isOpened():
            raise RuntimeError("Failed to create output writer")

        message_cb(f"Loading model {model_path}…")
        model = YOLO(model_path)
        try:
            if device in ("cuda", "cpu"):
                model.to(device)
        except Exception:
            pass

        allowed_ids = _resolve_allowed_ids(model.names, include_names)

        # ---------------------------------------------------------------
        # Read tracking configuration from env vars (Phase 1-2 improvements)
        # Defaults preserve existing behavior when env vars are not set.
        # ---------------------------------------------------------------
        track_min_conf = MOE_TRACK_MIN_CONF()
        track_min_area = MOE_TRACK_MIN_BOX_AREA()
        track_classes_str = MOE_TRACK_CLASSES()
        track_buffer = MOE_TRACK_BUFFER()
        track_thresh = MOE_TRACK_THRESH()
        track_match_thresh = MOE_TRACK_MATCH_THRESH()
        show_trails = MOE_SHOW_TRAILS()
        trail_len = MOE_TRAIL_LEN()
        
        # PASS 2: Professional outputs env vars
        show_hud = MOE_SHOW_HUD()
        write_csv_v2 = MOE_WRITE_CSV_V2()
        
        # Parse extra class filter from MOE_TRACK_CLASSES (comma-separated class IDs)
        extra_class_filter = None
        if track_classes_str.strip():
            try:
                extra_class_filter = set(int(x.strip()) for x in track_classes_str.split(",") if x.strip())
            except ValueError:
                extra_class_filter = None  # Invalid format, skip filtering

        # Create ByteTrack with configurable parameters
        # sv.ByteTrack accepts: track_activation_threshold, lost_track_buffer, 
        #                       minimum_matching_threshold, frame_rate
        tracker = sv.ByteTrack(
            track_activation_threshold=track_thresh,
            lost_track_buffer=track_buffer,
            minimum_matching_threshold=track_match_thresh,
            frame_rate=int(fps) if fps > 0 else 30
        )
        
        box_anno = sv.BoxAnnotator()
        label_anno = sv.LabelAnnotator(text_position=sv.Position.TOP_LEFT)

        last_pt: Dict[int, tuple] = {}
        ema_speed: Dict[int, float] = {}
        seen_by_class = defaultdict(set)   # cname -> {track_ids}
        seen_all_ids = set()
        
        # Trail storage: track_id -> deque of (cx, cy) for last N frames
        trails: Dict[int, deque] = defaultdict(lambda: deque(maxlen=trail_len))

        # PASS 2: Initialize TrackStats collector
        track_stats = TrackStats()
        
        # PASS 2: Prepare runs/offline output directory for summary.json and CSV v2
        # Use original_filename if provided, else fall back to src basename
        original_filename = job_rec.get("original_filename") or os.path.basename(src)
        video_basename = os.path.splitext(original_filename)[0]
        # Sanitize for filesystem (remove problematic chars)
        video_basename = "".join(c if c.isalnum() or c in "-_." else "_" for c in video_basename)
        runs_dir = os.path.join(os.getcwd(), "runs", "offline", video_basename)
        os.makedirs(runs_dir, exist_ok=True)
        summary_json_path = os.path.join(runs_dir, "summary.json")
        csv_v2_path = os.path.join(runs_dir, "tracks_v2.csv") if write_csv_v2 else None
        
        # PASS 2: CSV v2 file handle (opened only if enabled)
        csv_v2_file = None
        csv_v2_writer = None
        if write_csv_v2:
            try:
                csv_v2_file = open(csv_v2_path, "w", newline="")
                csv_v2_writer = csv.writer(csv_v2_file)
                csv_v2_writer.writerow([
                    "timestamp", "video", "frame", "track_id", "class_name",
                    "conf", "x1", "y1", "x2", "y2", "cx", "cy", "area"
                ])
            except Exception as e:
                message_cb(f"Warning: Could not open CSV v2 file: {e}")
                csv_v2_file = None
                csv_v2_writer = None
        
        # PASS 2: FPS tracking for HUD
        proc_start_time = time.time()
        proc_fps = 0.0

        with open(out_csv, "w", newline="") as fcsv:
            writer_csv = csv.writer(fcsv)
            writer_csv.writerow(["frame","track_id","class","cx","cy","speed_kmh"])

            message_cb("Processing frames…")
            frame_idx = 0
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                frame_idx += 1

                # inference
                y = model(frame, conf=conf, verbose=False)[0]
                det = sv.Detections.from_ultralytics(y)

                # Primary class filter (from job_rec 'include' parameter)
                if len(det) > 0:
                    if det.class_id is None or not len(det.class_id):
                        det = det[:0]
                    else:
                        keep = np.isin(det.class_id, list(allowed_ids))
                        det = det[keep]
                
                # Pre-tracking filters (Phase 2 improvement: reduces noisy detections)
                # Only applied if env vars are set; defaults preserve current behavior
                if len(det) > 0:
                    det = _filter_detections_pre_track(
                        det,
                        min_conf=track_min_conf,
                        min_area=track_min_area,
                        allowed_class_ids=extra_class_filter
                    )

                # tracking
                det = tracker.update_with_detections(det)

                # annotate + log
                labels = []
                now_t = frame_idx / fps
                active_this_frame = 0  # PASS 2: count for HUD
                
                for i in range(len(det)):
                    tid = int(det.tracker_id[i]) if det.tracker_id is not None else -1
                    cls_id = int(det.class_id[i]) if det.class_id is not None else -1
                    cname = str(model.names.get(cls_id, cls_id)) if isinstance(model.names, dict) else str(cls_id)
                    conf_score = float(det.confidence[i]) if det.confidence is not None else 0.0

                    cx, cy = _center_xyxy(det.xyxy[i])
                    box_ar = _box_area(det.xyxy[i])  # PASS 2: compute area for stats
                    
                    # PASS 2: Update tracklet statistics
                    if tid >= 0:
                        track_stats.update(tid, frame_idx, cname, conf_score, box_ar)
                        active_this_frame += 1
                    
                    # Update trail (if enabled)
                    if show_trails and tid >= 0:
                        trails[tid].append((int(cx), int(cy)))

                    spd_kmh = 0.0
                    if tid in last_pt:
                        px, py, pt = last_pt[tid]
                        dpx = float(np.hypot(cx - px, cy - py))
                        dt  = max(1e-6, now_t - pt)
                        mps = (dpx * meters_per_pixel) / dt
                        kmh = mps * 3.6
                        prev = ema_speed.get(tid, kmh)
                        spd_kmh = 0.2 * kmh + 0.8 * prev
                        ema_speed[tid] = spd_kmh
                    last_pt[tid] = (cx, cy, now_t)

                    # seen maps (note: counts are per raw track id)
                    seen_by_class[cname].add(tid)
                    seen_all_ids.add(tid)

                    # Clean ASCII-only label format (Phase 1 improvement)
                    labels.append(_format_label(cname, tid, conf_score, spd_kmh))
                    writer_csv.writerow([frame_idx, tid, cname, f"{cx:.2f}", f"{cy:.2f}", f"{spd_kmh:.3f}"])
                    
                    # PASS 2: Write to CSV v2 (if enabled)
                    if csv_v2_writer is not None:
                        x1, y1, x2, y2 = det.xyxy[i]
                        csv_v2_writer.writerow([
                            f"{now_t:.3f}",  # timestamp in seconds
                            video_basename,
                            frame_idx,
                            tid,
                            cname,
                            f"{conf_score:.4f}",
                            f"{x1:.2f}", f"{y1:.2f}", f"{x2:.2f}", f"{y2:.2f}",
                            f"{cx:.2f}", f"{cy:.2f}",
                            f"{box_ar:.2f}"
                        ])

                # Draw trails first (behind boxes) if enabled
                if show_trails:
                    _draw_trails(frame, trails, color=(0, 255, 255), thickness=2)
                
                frame = box_anno.annotate(frame, det)
                frame = label_anno.annotate(frame, det, labels=labels)
                
                # PASS 2: Draw HUD overlay (if enabled)
                if show_hud:
                    elapsed = time.time() - proc_start_time
                    proc_fps = frame_idx / max(elapsed, 0.001)
                    _draw_hud(frame, frame_idx, active_this_frame, 
                              track_stats.unique_tracks, proc_fps, tracking_enabled=True)
                
                writer.write(frame)

                if total_frames:
                    progress_cb(min(0.99, frame_idx / max(1, total_frames)))
                else:
                    progress_cb(min(0.99, (frame_idx % 2000) / 2000.0))

            message_cb("Finalizing…")

        cap.release()
        writer.release()
        
        # Re-encode video for browser compatibility (Chrome, Firefox, Edge)
        message_cb("Re-encoding video for browser compatibility...")
        if reencode_video_for_browser(out_video):
            message_cb("Video re-encoded to H.264")
        else:
            message_cb("Note: ffmpeg not available, video may not play in Chrome/Firefox")
        
        # PASS 2: Close CSV v2 file if opened
        if csv_v2_file is not None:
            try:
                csv_v2_file.close()
            except Exception:
                pass

        # Write JSON summary (per-class unique track counts) - existing behavior
        if out_json:
            summary = {
                "fps": float(fps),
                "frames": int(frame_idx),
                "unique_objects_total": int(len(seen_all_ids)),
                "unique_objects_by_class": {k: int(len(v)) for k, v in sorted(seen_by_class.items())}
            }
            with open(out_json, "w") as jf:
                json.dump(summary, jf, indent=2)
        
        # PASS 2: Write professional summary.json with per-tracklet stats
        # Always write to runs/offline/<video>/ directory
        track_summary = track_stats.export_summary(video_basename, fps, frame_idx)
        _write_summary_json_safe(track_summary, summary_json_path, message_cb)

        progress_cb(1.0)
        message_cb("Done.")

    except Exception as e:
        job_rec["error"] = str(e)
        progress_cb(-1.0)
        message_cb(f"Error: {e}")
