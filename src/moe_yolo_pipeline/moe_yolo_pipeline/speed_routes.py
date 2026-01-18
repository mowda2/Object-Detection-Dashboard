"""
Speed (Calibrated) Routes - v2 with settings and library integration

This blueprint provides video upload, homography calibration, and
calibrated speed measurement with configurable settings.

This module is completely isolated and does NOT interfere with existing
offline analysis or Roboflow pipelines.

Routes:
    GET  /speed                     - Main page with upload form + settings
    POST /speed/upload              - Handle video upload, extract calibration frame
    GET  /speed/calibrate/<job_id>  - Calibration form page
    POST /speed/calibrate/<job_id>  - Save homography calibration
    GET  /speed/run/<job_id>        - Run speed analysis
    GET  /speed/results/<job_id>    - Show analysis results
    GET  /speed/artifact/<job_id>/<filename> - Serve job artifacts

v2 Improvements:
- Configurable settings via UI (saved to settings.json)
- Library integration (analyses recorded in shared library)
- Cache fingerprinting for reproducible reruns
"""

import os
import uuid
import json
import hashlib
from datetime import datetime
from typing import Optional, Dict, Any, List, Tuple

from flask import (
    Blueprint,
    render_template,
    request,
    jsonify,
    redirect,
    url_for,
    send_file,
    abort,
)

import cv2
import numpy as np


# Blueprint for Speed routes
speed_bp = Blueprint(
    "speed",
    __name__,
    template_folder="templates",
    url_prefix="/speed",
)

# Directory for speed jobs (under project root runs/speed/)
# This keeps outputs separate from the package directory
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
SPEED_JOBS_DIR = os.path.join(PROJECT_ROOT, "runs", "speed")

os.makedirs(SPEED_JOBS_DIR, exist_ok=True)

# Allowed video extensions and size limit
ALLOWED_EXTENSIONS = {".mp4", ".mov"}
MAX_FILE_SIZE = 500 * 1024 * 1024  # 500MB limit


def _allowed_file(filename: str) -> bool:
    """Check if file extension is allowed."""
    ext = os.path.splitext(filename)[1].lower()
    return ext in ALLOWED_EXTENSIONS


def _get_job_dir(job_id: str) -> str:
    """Get the directory for a job."""
    return os.path.join(SPEED_JOBS_DIR, job_id)


def _load_calibration(job_id: str) -> Optional[Dict[str, Any]]:
    """Load calibration data from disk if it exists."""
    cal_path = os.path.join(_get_job_dir(job_id), "calibration.json")
    if not os.path.exists(cal_path):
        return None
    try:
        with open(cal_path, "r") as f:
            return json.load(f)
    except Exception:
        return None


def _save_calibration(job_id: str, data: Dict[str, Any]) -> bool:
    """Save calibration data to disk."""
    job_dir = _get_job_dir(job_id)
    cal_path = os.path.join(job_dir, "calibration.json")
    try:
        os.makedirs(job_dir, exist_ok=True)
        with open(cal_path, "w") as f:
            json.dump(data, f, indent=2)
        return True
    except Exception:
        return False


def _load_settings(job_id: str) -> Optional[Dict[str, Any]]:
    """Load settings from disk if they exist."""
    settings_path = os.path.join(_get_job_dir(job_id), "settings.json")
    if not os.path.exists(settings_path):
        return None
    try:
        with open(settings_path, "r") as f:
            return json.load(f)
    except Exception:
        return None


def _save_settings(job_id: str, settings: Dict[str, Any]) -> bool:
    """Save settings to disk."""
    job_dir = _get_job_dir(job_id)
    settings_path = os.path.join(job_dir, "settings.json")
    try:
        os.makedirs(job_dir, exist_ok=True)
        with open(settings_path, "w") as f:
            json.dump(settings, f, indent=2)
        return True
    except Exception:
        return False


def _parse_settings_from_form(form) -> Dict[str, Any]:
    """Parse settings from form submission."""
    from .speed_analyzer import get_default_settings
    
    settings = get_default_settings()
    
    # Detection settings
    if form.get("min_conf"):
        try:
            settings["min_conf"] = float(form["min_conf"])
        except ValueError:
            pass
    
    if form.get("max_det_per_frame"):
        try:
            settings["max_det_per_frame"] = int(form["max_det_per_frame"])
        except ValueError:
            pass
    
    if form.get("class_filter"):
        settings["class_filter"] = form["class_filter"].strip()
    
    if form.get("min_box_area"):
        try:
            settings["min_box_area"] = int(form["min_box_area"])
        except ValueError:
            pass
    
    # Aspect ratio filter
    settings["aspect_ratio_filter"] = form.get("aspect_ratio_filter") == "on"
    if form.get("aspect_ratio_min"):
        try:
            settings["aspect_ratio_min"] = float(form["aspect_ratio_min"])
        except ValueError:
            pass
    if form.get("aspect_ratio_max"):
        try:
            settings["aspect_ratio_max"] = float(form["aspect_ratio_max"])
        except ValueError:
            pass
    
    # ROI filter
    settings["roi_enabled"] = form.get("roi_enabled") == "on"
    for key in ["roi_x1", "roi_y1", "roi_x2", "roi_y2"]:
        if form.get(key):
            try:
                settings[key] = int(form[key])
            except ValueError:
                pass
    
    # Tracking settings
    if form.get("track_thresh"):
        try:
            settings["track_thresh"] = float(form["track_thresh"])
        except ValueError:
            pass
    if form.get("match_thresh"):
        try:
            settings["match_thresh"] = float(form["match_thresh"])
        except ValueError:
            pass
    if form.get("track_buffer"):
        try:
            settings["track_buffer"] = int(form["track_buffer"])
        except ValueError:
            pass
    
    # Speed settings
    if form.get("smoothing_window_s"):
        try:
            settings["smoothing_window_s"] = float(form["smoothing_window_s"])
        except ValueError:
            pass
    if form.get("max_kph"):
        try:
            settings["max_kph"] = float(form["max_kph"])
        except ValueError:
            pass
    if form.get("min_dt_s"):
        try:
            settings["min_dt_s"] = float(form["min_dt_s"])
        except ValueError:
            pass
    if form.get("outlier_reject_kph_per_s"):
        try:
            settings["outlier_reject_kph_per_s"] = float(form["outlier_reject_kph_per_s"])
        except ValueError:
            pass
    if form.get("median_filter_n"):
        try:
            settings["median_filter_n"] = int(form["median_filter_n"])
        except ValueError:
            pass
    if form.get("jump_reject_m"):
        try:
            settings["jump_reject_m"] = float(form["jump_reject_m"])
        except ValueError:
            pass
    
    # Performance settings
    if form.get("frame_skip"):
        try:
            settings["frame_skip"] = max(1, int(form["frame_skip"]))
        except ValueError:
            pass
    if form.get("resize_width"):
        try:
            settings["resize_width"] = max(0, int(form["resize_width"]))
        except ValueError:
            pass
    
    # Trail visualization settings
    settings["enable_trails"] = form.get("enable_trails") == "on"
    if form.get("trail_seconds"):
        try:
            settings["trail_seconds"] = max(0.1, min(10.0, float(form["trail_seconds"])))
        except ValueError:
            pass
    if form.get("trail_max_points"):
        try:
            settings["trail_max_points"] = max(2, min(200, int(form["trail_max_points"])))
        except ValueError:
            pass
    if form.get("trail_stride"):
        try:
            settings["trail_stride"] = max(1, min(10, int(form["trail_stride"])))
        except ValueError:
            pass
    if form.get("trail_thickness"):
        try:
            settings["trail_thickness"] = max(1, min(10, int(form["trail_thickness"])))
        except ValueError:
            pass
    if form.get("trail_alpha"):
        try:
            settings["trail_alpha"] = max(0.1, min(1.0, float(form["trail_alpha"])))
        except ValueError:
            pass
    settings["trail_fade"] = form.get("trail_fade") == "on" if "trail_fade" in form else settings.get("trail_fade", True)
    if form.get("trail_anchor"):
        if form["trail_anchor"] in ("bottom_center", "center"):
            settings["trail_anchor"] = form["trail_anchor"]
    
    # ---------------------------------------------------------------------------
    # Violation detection settings
    # ---------------------------------------------------------------------------
    settings["violation_enabled"] = form.get("violation_enabled") == "on"
    if form.get("violation_speed_kmh"):
        try:
            settings["violation_speed_kmh"] = max(10.0, min(300.0, float(form["violation_speed_kmh"])))
        except ValueError:
            pass
    if form.get("violation_min_seconds"):
        try:
            settings["violation_min_seconds"] = max(0.1, min(5.0, float(form["violation_min_seconds"])))
        except ValueError:
            pass
    if form.get("violation_cooldown_seconds"):
        try:
            settings["violation_cooldown_seconds"] = max(0.5, min(30.0, float(form["violation_cooldown_seconds"])))
        except ValueError:
            pass
    if form.get("violation_capture_mode"):
        if form["violation_capture_mode"] in ("first_crossing", "peak_speed"):
            settings["violation_capture_mode"] = form["violation_capture_mode"]
    settings["violation_save_full_frame"] = form.get("violation_save_full_frame") == "on" if "violation_save_full_frame" in form else settings.get("violation_save_full_frame", True)
    settings["violation_save_crop"] = form.get("violation_save_crop") == "on" if "violation_save_crop" in form else settings.get("violation_save_crop", True)
    if form.get("violation_crop_padding_px"):
        try:
            settings["violation_crop_padding_px"] = max(0, min(100, int(form["violation_crop_padding_px"])))
        except ValueError:
            pass
    
    return settings


def _extract_first_frame(video_path: str, output_path: str) -> bool:
    """
    Extract the first frame from a video and save as JPEG.
    
    Returns True on success, False on failure.
    """
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return False
        
        ret, frame = cap.read()
        cap.release()
        
        if not ret or frame is None:
            return False
        
        # Save as JPEG with good quality
        cv2.imwrite(output_path, frame, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
        return os.path.exists(output_path)
    except Exception:
        return False


def _compute_homography(pixel_pts: List[List[float]], 
                        world_pts: List[List[float]]) -> Optional[np.ndarray]:
    """
    Compute homography matrix H that maps pixel coordinates to world coordinates.
    
    Args:
        pixel_pts: List of 4 [u, v] pixel coordinates
        world_pts: List of 4 [X, Y] world coordinates in meters
    
    Returns:
        3x3 homography matrix H, or None if computation fails
    """
    try:
        src = np.array(pixel_pts, dtype=np.float32)
        dst = np.array(world_pts, dtype=np.float32)
        
        if src.shape != (4, 2) or dst.shape != (4, 2):
            return None
        
        # Use getPerspectiveTransform for exactly 4 points (more stable)
        H = cv2.getPerspectiveTransform(src, dst)
        
        # Validate H is not degenerate
        if H is None:
            return None
        
        # Check determinant is reasonable (not near-zero)
        det = np.linalg.det(H)
        if abs(det) < 1e-10:
            return None
        
        return H
    except Exception:
        return None


def _compute_calibration_sanity(pixel_pts: List[List[float]], 
                                 world_pts: List[List[float]]) -> Dict[str, Any]:
    """
    Compute sanity check metrics for calibration.
    
    Returns warnings if calibration seems problematic.
    """
    result = {"warnings": [], "metrics": {}}
    
    try:
        # Compute world distances
        p1, p2, p3, p4 = world_pts
        width_top = np.sqrt((p2[0]-p1[0])**2 + (p2[1]-p1[1])**2)
        width_bottom = np.sqrt((p3[0]-p4[0])**2 + (p3[1]-p4[1])**2)
        height_left = np.sqrt((p4[0]-p1[0])**2 + (p4[1]-p1[1])**2)
        height_right = np.sqrt((p3[0]-p2[0])**2 + (p3[1]-p2[1])**2)
        
        result["metrics"]["width_top_m"] = round(width_top, 2)
        result["metrics"]["width_bottom_m"] = round(width_bottom, 2)
        result["metrics"]["height_left_m"] = round(height_left, 2)
        result["metrics"]["height_right_m"] = round(height_right, 2)
        
        # Check for very small calibration area
        avg_width = (width_top + width_bottom) / 2
        avg_height = (height_left + height_right) / 2
        area_m2 = avg_width * avg_height
        result["metrics"]["approx_area_m2"] = round(area_m2, 2)
        
        if area_m2 < 10:
            result["warnings"].append(
                f"Calibration area is small ({area_m2:.1f} m²). "
                "Consider using a larger rectangle for better accuracy."
            )
        
        # Check for very large calibration area (likely wrong units)
        if area_m2 > 50000:
            result["warnings"].append(
                f"Calibration area is very large ({area_m2:.0f} m²). "
                "Check that world coordinates are in meters, not other units."
            )
        
        # Check for significantly non-rectangular shape
        width_ratio = max(width_top, width_bottom) / max(0.01, min(width_top, width_bottom))
        height_ratio = max(height_left, height_right) / max(0.01, min(height_left, height_right))
        
        if width_ratio > 2.0 or height_ratio > 2.0:
            result["warnings"].append(
                "Calibration rectangle appears significantly skewed. "
                "This may affect accuracy at certain positions."
            )
        
    except Exception as e:
        result["warnings"].append(f"Could not compute sanity check: {e}")
    
    return result


# ---------------------------------------------------------------------------
# Library Integration (backward-compatible)
# ---------------------------------------------------------------------------

def _register_to_library(job_id: str, summary: Dict[str, Any], status: str = "completed",
                         error_msg: str = None) -> bool:
    """
    Register a speed analysis job to the shared Library system.
    
    This adds a new entry without breaking existing library behavior.
    The library uses SQLite (library_db.py) for the offline module,
    so we need to integrate carefully.
    """
    try:
        # Import the library database
        from .library_db import LibraryDB
        
        # Use the same DB path as offline module
        pkg_dir = os.path.dirname(__file__)
        data_dir = os.path.join(pkg_dir, "data")
        db_path = os.path.join(data_dir, "library.sqlite3")
        
        lib = LibraryDB(db_path)
        
        job_dir = _get_job_dir(job_id)
        
        # Compute file hash for the input video
        input_path = os.path.join(job_dir, "input.mp4")
        if os.path.exists(input_path):
            file_hash = _sha256_file(input_path)
            file_size = os.path.getsize(input_path)
            
            # Upsert the file record
            lib.upsert_file(
                file_hash=file_hash,
                filename=summary.get("video", "input.mp4"),
                size_bytes=file_size,
                stored_path=input_path
            )
        else:
            file_hash = hashlib.sha256(job_id.encode()).hexdigest()
        
        # Create a params_hash that includes speed-specific settings
        settings_used = summary.get("settings_used", {})
        params_str = json.dumps({
            "source_page": "speed",
            "analysis_type": "speed_calibrated",
            "settings_fingerprint": summary.get("settings_fingerprint", ""),
        }, sort_keys=True)
        params_hash = hashlib.sha256(params_str.encode()).hexdigest()
        
        # Build analysis record (compatible with existing schema)
        # Note: existing schema has model, include, conf, mpp, device fields
        # We repurpose some for speed-specific data
        analysis_rec = {
            "id": job_id,
            "file_hash": file_hash,
            "params_hash": params_hash,
            "model": "speed_calibrated",  # Use as analysis type marker
            "include": "speed",  # Use as source_page marker
            "conf": settings_used.get("min_conf", 0.25),
            "mpp": 0.0,  # Not used for speed
            "device": "cpu",
            "fps": summary.get("fps", 30.0),
            "frames": summary.get("frames_processed", 0),
            "video_path": os.path.join(job_dir, "output.mp4"),
            "csv_path": os.path.join(job_dir, "tracks_v1.csv"),
            "json_path": os.path.join(job_dir, "summary.json"),
            "poster_path": os.path.join(job_dir, "calibration_frame.jpg"),
        }
        
        # Build class stats (track counts as pseudo-class stats)
        class_counts = {}
        per_track = summary.get("per_track", {})
        for tid, track_data in per_track.items():
            cls_name = track_data.get("class", "unknown")
            class_counts[cls_name] = class_counts.get(cls_name, 0) + 1
        
        # Insert analysis record
        lib.insert_analysis(analysis_rec, class_counts)
        
        return True
        
    except Exception as e:
        # Don't fail the analysis if library integration fails
        print(f"Warning: Failed to register to library: {e}")
        return False


def _sha256_file(path: str, buf: int = 1024*1024) -> str:
    """Compute SHA256 hash of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(buf)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


# ============== Routes ==============

@speed_bp.route("/")
def speed_index():
    """Main Speed page with upload form and settings."""
    from .speed_analyzer import get_default_settings
    defaults = get_default_settings()
    return render_template("speed/index.html", defaults=defaults)


@speed_bp.route("/upload", methods=["POST"])
def speed_upload():
    """Handle video upload, extract calibration frame, and save settings."""
    from .speed_analyzer import get_default_settings
    defaults = get_default_settings()
    
    # Check for file
    if "file" not in request.files:
        return render_template("speed/index.html", error="No file uploaded", defaults=defaults), 400
    
    file = request.files["file"]
    if not file.filename:
        return render_template("speed/index.html", error="Empty filename", defaults=defaults), 400
    
    if not _allowed_file(file.filename):
        return render_template(
            "speed/index.html",
            error=f"Invalid file type. Allowed: {', '.join(ALLOWED_EXTENSIONS)}",
            defaults=defaults
        ), 400
    
    # Generate job ID
    job_id = uuid.uuid4().hex[:12]
    job_dir = _get_job_dir(job_id)
    os.makedirs(job_dir, exist_ok=True)
    
    # Save uploaded video
    video_path = os.path.join(job_dir, "input.mp4")
    try:
        file.save(video_path)
    except Exception as e:
        return render_template("speed/index.html", error=f"Failed to save file: {e}", defaults=defaults), 500
    
    # Check file size
    file_size = os.path.getsize(video_path)
    if file_size > MAX_FILE_SIZE:
        os.remove(video_path)
        return render_template(
            "speed/index.html",
            error=f"File too large. Maximum: {MAX_FILE_SIZE // (1024*1024)}MB",
            defaults=defaults
        ), 400
    
    # Extract first frame for calibration
    frame_path = os.path.join(job_dir, "calibration_frame.jpg")
    if not _extract_first_frame(video_path, frame_path):
        # Clean up on failure
        if os.path.exists(video_path):
            os.remove(video_path)
        return render_template(
            "speed/index.html",
            error="Failed to extract frame from video. Is the video valid?",
            defaults=defaults
        ), 400
    
    # Parse and save settings from form
    settings = _parse_settings_from_form(request.form)
    _save_settings(job_id, settings)
    
    # Redirect to calibration page
    return redirect(url_for("speed.speed_calibrate", job_id=job_id))


@speed_bp.route("/calibrate/<job_id>", methods=["GET"])
def speed_calibrate(job_id: str):
    """Calibration form page - show frame and point entry form."""
    job_dir = _get_job_dir(job_id)
    
    # Verify job exists
    frame_path = os.path.join(job_dir, "calibration_frame.jpg")
    if not os.path.exists(frame_path):
        abort(404, description="Job not found or calibration frame missing")
    
    # Check if calibration already exists
    existing_cal = _load_calibration(job_id)
    
    # Load existing settings
    existing_settings = _load_settings(job_id)
    
    # Compute sanity check if calibration exists
    sanity_check = None
    if existing_cal:
        sanity_check = _compute_calibration_sanity(
            existing_cal.get("pixel_points", []),
            existing_cal.get("world_points_m", [])
        )
    
    # Get frame dimensions for display info
    frame_info = {}
    try:
        img = cv2.imread(frame_path)
        if img is not None:
            frame_info["height"], frame_info["width"] = img.shape[:2]
    except Exception:
        pass
    
    return render_template(
        "speed/calibrate.html",
        job_id=job_id,
        frame_info=frame_info,
        existing_cal=existing_cal,
        existing_settings=existing_settings,
        sanity_check=sanity_check,
        success=request.args.get("success"),
    )


@speed_bp.route("/calibrate/<job_id>", methods=["POST"])
def speed_calibrate_submit(job_id: str):
    """Process calibration form submission."""
    job_dir = _get_job_dir(job_id)
    
    # Verify job exists
    video_path = os.path.join(job_dir, "input.mp4")
    if not os.path.exists(video_path):
        abort(404, description="Job not found")
    
    # Parse pixel points from form
    try:
        pixel_pts = [
            [float(request.form["p1_u"]), float(request.form["p1_v"])],
            [float(request.form["p2_u"]), float(request.form["p2_v"])],
            [float(request.form["p3_u"]), float(request.form["p3_v"])],
            [float(request.form["p4_u"]), float(request.form["p4_v"])],
        ]
    except (KeyError, ValueError) as e:
        return render_template(
            "speed/calibrate.html",
            job_id=job_id,
            error=f"Invalid pixel coordinates: {e}",
            existing_cal=_load_calibration(job_id),
        ), 400
    
    # Parse world points from form
    try:
        world_pts = [
            [float(request.form["P1_X"]), float(request.form["P1_Y"])],
            [float(request.form["P2_X"]), float(request.form["P2_Y"])],
            [float(request.form["P3_X"]), float(request.form["P3_Y"])],
            [float(request.form["P4_X"]), float(request.form["P4_Y"])],
        ]
    except (KeyError, ValueError) as e:
        return render_template(
            "speed/calibrate.html",
            job_id=job_id,
            error=f"Invalid world coordinates: {e}",
            existing_cal=_load_calibration(job_id),
        ), 400
    
    # Compute homography
    H = _compute_homography(pixel_pts, world_pts)
    if H is None:
        return render_template(
            "speed/calibrate.html",
            job_id=job_id,
            error="Failed to compute homography. Check that points form a valid quadrilateral.",
            existing_cal=_load_calibration(job_id),
        ), 400
    
    # Save calibration data
    calibration_data = {
        "video_path": "input.mp4",
        "pixel_points": pixel_pts,
        "world_points_m": world_pts,
        "H_pixel_to_world": H.tolist(),
    }
    
    if not _save_calibration(job_id, calibration_data):
        return render_template(
            "speed/calibrate.html",
            job_id=job_id,
            error="Failed to save calibration data.",
            existing_cal=_load_calibration(job_id),
        ), 500
    
    # Redirect with success flag
    return redirect(url_for("speed.speed_calibrate", job_id=job_id, success="1"))


@speed_bp.route("/artifact/<job_id>/<path:filename>")
def speed_artifact(job_id: str, filename: str):
    """Serve job artifacts (calibration frame, output video, CSV, etc.)."""
    # Security: only allow specific filenames to prevent path traversal
    allowed_files = {
        "calibration_frame.jpg",
        "input.mp4",
        "calibration.json",
        "settings.json",
        "output.mp4",
        "tracks_v1.csv",
        "summary.json",
    }
    
    # Check for violations subdirectory files
    is_violation_file = False
    if filename.startswith("violations/"):
        # Allow violations/*.jpg, violations.json, violations.csv
        sub_filename = filename[len("violations/"):]
        if (sub_filename in ("violations.json", "violations.csv") or 
            (sub_filename.startswith(("full_", "crop_")) and sub_filename.endswith(".jpg"))):
            is_violation_file = True
    
    if filename not in allowed_files and not is_violation_file:
        abort(403, description="Access to this file is not allowed")
    
    job_dir = _get_job_dir(job_id)
    file_path = os.path.join(job_dir, filename)
    
    if not os.path.exists(file_path):
        abort(404, description="File not found")
    
    # Determine mimetype and download behavior
    if filename.endswith(".jpg"):
        mimetype = "image/jpeg"
    elif filename.endswith(".mp4"):
        mimetype = "video/mp4"
    elif filename.endswith(".json"):
        mimetype = "application/json"
    elif filename.endswith(".csv"):
        mimetype = "text/csv"
    else:
        mimetype = "application/octet-stream"
    
    # For CSV, set as attachment for download
    as_attachment = filename.endswith(".csv")
    
    return send_file(file_path, mimetype=mimetype, as_attachment=as_attachment)


def _load_summary(job_id: str) -> Optional[Dict[str, Any]]:
    """Load summary.json for a job if it exists."""
    summary_path = os.path.join(_get_job_dir(job_id), "summary.json")
    if not os.path.exists(summary_path):
        return None
    try:
        with open(summary_path, "r") as f:
            return json.load(f)
    except Exception:
        return None


def _load_violations(job_id: str) -> Optional[List[Dict[str, Any]]]:
    """Load violations.json for a job if it exists."""
    violations_path = os.path.join(_get_job_dir(job_id), "violations", "violations.json")
    if not os.path.exists(violations_path):
        return None
    try:
        with open(violations_path, "r") as f:
            return json.load(f)
    except Exception:
        return None


def _job_has_results(job_id: str) -> bool:
    """Check if job has completed analysis results."""
    job_dir = _get_job_dir(job_id)
    return (
        os.path.exists(os.path.join(job_dir, "output.mp4")) and
        os.path.exists(os.path.join(job_dir, "tracks_v1.csv")) and
        os.path.exists(os.path.join(job_dir, "summary.json"))
    )


@speed_bp.route("/run/<job_id>")
def speed_run(job_id: str):
    """Run speed analysis on a calibrated job."""
    job_dir = _get_job_dir(job_id)
    
    # Verify job exists
    if not os.path.exists(os.path.join(job_dir, "input.mp4")):
        abort(404, description="Job not found")
    
    # Check if calibration exists
    if not os.path.exists(os.path.join(job_dir, "calibration.json")):
        return render_template(
            "speed/error.html",
            job_id=job_id,
            error="Calibration not found. Please calibrate first.",
            back_url=url_for("speed.speed_calibrate", job_id=job_id),
        )
    
    # Load settings (use defaults if not found)
    settings = _load_settings(job_id)
    
    # Check if results already exist with matching settings
    summary_path = os.path.join(job_dir, "summary.json")
    if _job_has_results(job_id) and os.path.exists(summary_path):
        try:
            with open(summary_path, "r") as f:
                existing_summary = json.load(f)
            # Check if settings match (use fingerprint)
            from .speed_analyzer import _settings_hash, get_default_settings
            current_settings = settings or get_default_settings()
            current_fingerprint = _settings_hash(current_settings)
            cached_fingerprint = existing_summary.get("settings_fingerprint", "")
            
            if current_fingerprint == cached_fingerprint:
                return redirect(url_for("speed.speed_results", job_id=job_id))
        except Exception:
            pass
    
    # Run analysis
    try:
        # Import here to avoid circular imports and keep isolation
        from .speed_analyzer import run_speed_job
        
        summary = run_speed_job(job_dir, settings=settings)
        
        # Register to library on success
        _register_to_library(job_id, summary, status="completed")
        
        return redirect(url_for("speed.speed_results", job_id=job_id))
    
    except Exception as e:
        # Register failure to library
        _register_to_library(
            job_id,
            {"video": "input.mp4", "error": str(e)},
            status="failed",
            error_msg=str(e)
        )
        
        return render_template(
            "speed/error.html",
            job_id=job_id,
            error=f"Analysis failed: {str(e)}",
            back_url=url_for("speed.speed_calibrate", job_id=job_id),
        )


@speed_bp.route("/results/<job_id>")
def speed_results(job_id: str):
    """Show analysis results for a job."""
    job_dir = _get_job_dir(job_id)
    
    # Verify results exist
    if not _job_has_results(job_id):
        return redirect(url_for("speed.speed_run", job_id=job_id))
    
    # Load summary
    summary = _load_summary(job_id)
    if summary is None:
        abort(500, description="Failed to load analysis summary")
    
    # Load violations if available
    violations = _load_violations(job_id)
    
    return render_template(
        "speed/results.html",
        job_id=job_id,
        summary=summary,
        violations=violations,
    )
