"""
Roboflow Hosted Inference Routes

This blueprint provides a completely isolated Roboflow inference feature.
It does NOT interfere with the existing offline analysis pipeline.

Routes:
    GET  /roboflow              - Main page with upload form
    POST /roboflow/upload       - Handle video upload and start job
    GET  /roboflow/job/<jid>    - Job viewer page
    GET  /roboflow/status/<jid> - Job status (JSON)
    GET  /roboflow/frame/<jid>/<idx> - Get annotated frame image
    GET  /roboflow/detections/<jid>  - Get all detections (JSON)
"""

import os
import uuid
import json
import threading
import time
import csv
from typing import Dict, Any, Optional, List
from dataclasses import asdict

from flask import (
    Blueprint,
    render_template,
    request,
    jsonify,
    send_file,
    abort,
    current_app,
)

import cv2
import numpy as np

from .roboflow_client import (
    RoboflowClient,
    RoboflowClientError,
    RoboflowConfigError,
    RoboflowAPIError,
    Detection,
    draw_detections,
    get_client,
)


# Blueprint for Roboflow routes
roboflow_bp = Blueprint(
    "roboflow",
    __name__,
    template_folder="templates",
    url_prefix="/roboflow",
)

# Directory for Roboflow jobs (separate from offline jobs)
PKG_DIR = os.path.dirname(__file__)
ROBOFLOW_DIR = os.path.join(PKG_DIR, "roboflow_jobs")
UPLOADS_DIR = os.path.join(ROBOFLOW_DIR, "uploads")
JOBS_DIR = os.path.join(ROBOFLOW_DIR, "jobs")

os.makedirs(UPLOADS_DIR, exist_ok=True)
os.makedirs(JOBS_DIR, exist_ok=True)

# In-memory job tracking (for active jobs only)
ACTIVE_JOBS: Dict[str, Dict[str, Any]] = {}

# Allowed video extensions
ALLOWED_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}
MAX_FILE_SIZE = 500 * 1024 * 1024  # 500MB limit for Roboflow jobs


def _allowed_file(filename: str) -> bool:
    """Check if file extension is allowed."""
    ext = os.path.splitext(filename)[1].lower()
    return ext in ALLOWED_EXTENSIONS


def _safe_filename(filename: str) -> str:
    """Create a safe filename."""
    # Keep only alphanumeric, dash, underscore, dot
    safe = "".join(c if c.isalnum() or c in "-_." else "_" for c in filename)
    return safe[:100]  # Limit length


def _get_job_dir(job_id: str) -> str:
    """Get the directory for a job."""
    return os.path.join(JOBS_DIR, job_id)


def _get_job_meta_path(job_id: str) -> str:
    """Get path to job metadata file."""
    return os.path.join(_get_job_dir(job_id), "meta.json")


def _load_job_meta(job_id: str) -> Optional[Dict[str, Any]]:
    """Load job metadata from disk."""
    meta_path = _get_job_meta_path(job_id)
    if not os.path.exists(meta_path):
        return None
    try:
        with open(meta_path, "r") as f:
            return json.load(f)
    except Exception:
        return None


def _save_job_meta(job_id: str, meta: Dict[str, Any]):
    """Save job metadata to disk."""
    meta_path = _get_job_meta_path(job_id)
    os.makedirs(os.path.dirname(meta_path), exist_ok=True)
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)


def _process_video_job(job_id: str, video_path: str, frame_skip: int, confidence: float, overlap: float):
    """
    Background worker to process a video with Roboflow API.
    
    This function runs in a separate thread and updates job status.
    """
    job_dir = _get_job_dir(job_id)
    frames_dir = os.path.join(job_dir, "frames")
    os.makedirs(frames_dir, exist_ok=True)
    
    detections_path = os.path.join(job_dir, "detections.jsonl")
    csv_path = os.path.join(job_dir, "detections.csv")
    
    # Initialize client with job-specific settings
    client = RoboflowClient(confidence=confidence, overlap=overlap)
    
    if not client.is_configured():
        ACTIVE_JOBS[job_id]["state"] = "error"
        ACTIVE_JOBS[job_id]["error"] = "Roboflow not configured: " + ", ".join(client.get_missing_config())
        _save_job_meta(job_id, ACTIVE_JOBS[job_id])
        return
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        ACTIVE_JOBS[job_id]["state"] = "error"
        ACTIVE_JOBS[job_id]["error"] = "Failed to open video file"
        _save_job_meta(job_id, ACTIVE_JOBS[job_id])
        return
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    frames_to_process = (total_frames + frame_skip - 1) // frame_skip
    estimated_calls = client.estimate_api_calls(total_frames, frame_skip)
    
    ACTIVE_JOBS[job_id].update({
        "total_frames": total_frames,
        "frames_to_process": frames_to_process,
        "estimated_api_calls": estimated_calls,
        "fps": fps,
        "width": width,
        "height": height,
        "processed_frames": 0,
        "api_calls_made": 0,
    })
    _save_job_meta(job_id, ACTIVE_JOBS[job_id])
    
    # Process frames
    frame_idx = 0
    processed_count = 0
    all_detections = []
    
    # Open CSV writer
    with open(csv_path, "w", newline="") as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(["frame", "class", "confidence", "x1", "y1", "x2", "y2"])
        
        # Open JSONL for detections
        with open(detections_path, "w") as jsonl_file:
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Check if we should process this frame
                if frame_idx % frame_skip == 0:
                    try:
                        # Call Roboflow API
                        detections, metadata = client.infer_image(frame)
                        
                        ACTIVE_JOBS[job_id]["api_calls_made"] += 1
                        
                        # Draw and save annotated frame
                        annotated = draw_detections(frame, detections, box_color=(0, 200, 100))
                        frame_path = os.path.join(frames_dir, f"frame_{frame_idx:06d}.jpg")
                        cv2.imwrite(frame_path, annotated, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
                        
                        # Log detections
                        frame_dets = {
                            "frame": frame_idx,
                            "detections": [d.to_dict() for d in detections],
                        }
                        jsonl_file.write(json.dumps(frame_dets) + "\n")
                        
                        # Write to CSV
                        for det in detections:
                            csv_writer.writerow([
                                frame_idx,
                                det.class_name,
                                f"{det.confidence:.3f}",
                                f"{det.x1:.1f}",
                                f"{det.y1:.1f}",
                                f"{det.x2:.1f}",
                                f"{det.y2:.1f}",
                            ])
                        
                        all_detections.append(frame_dets)
                        processed_count += 1
                        
                        # Update progress
                        progress = processed_count / max(1, frames_to_process)
                        ACTIVE_JOBS[job_id]["processed_frames"] = processed_count
                        ACTIVE_JOBS[job_id]["progress"] = progress
                        ACTIVE_JOBS[job_id]["current_frame"] = frame_idx
                        ACTIVE_JOBS[job_id]["message"] = f"Processing frame {frame_idx}/{total_frames}"
                        
                    except RoboflowAPIError as e:
                        # Log error but continue if possible
                        ACTIVE_JOBS[job_id]["last_error"] = str(e)
                        # Save raw frame on error
                        frame_path = os.path.join(frames_dir, f"frame_{frame_idx:06d}.jpg")
                        cv2.imwrite(frame_path, frame, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
                        processed_count += 1
                    
                    except Exception as e:
                        ACTIVE_JOBS[job_id]["state"] = "error"
                        ACTIVE_JOBS[job_id]["error"] = str(e)
                        cap.release()
                        _save_job_meta(job_id, ACTIVE_JOBS[job_id])
                        return
                
                frame_idx += 1
    
    cap.release()
    
    # Save summary
    summary = {
        "total_frames": total_frames,
        "processed_frames": processed_count,
        "api_calls_made": ACTIVE_JOBS[job_id].get("api_calls_made", 0),
        "frame_skip": frame_skip,
        "fps": fps,
        "width": width,
        "height": height,
        "model": client.model,
    }
    
    summary_path = os.path.join(job_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    
    # Mark complete
    ACTIVE_JOBS[job_id]["state"] = "done"
    ACTIVE_JOBS[job_id]["progress"] = 1.0
    ACTIVE_JOBS[job_id]["message"] = "Processing complete"
    _save_job_meta(job_id, ACTIVE_JOBS[job_id])


# ============== Routes ==============

@roboflow_bp.route("/")
def roboflow_index():
    """Main Roboflow page with upload form."""
    client = get_client()
    config_status = client.get_config_status()
    missing_config = client.get_missing_config()
    
    return render_template(
        "roboflow/index.html",
        config_status=config_status,
        missing_config=missing_config,
        is_configured=client.is_configured(),
    )


@roboflow_bp.route("/upload", methods=["POST"])
def roboflow_upload():
    """Handle video upload and start processing job."""
    client = get_client()
    
    if not client.is_configured():
        return jsonify({
            "ok": False,
            "error": "Roboflow not configured",
            "missing": client.get_missing_config(),
        }), 400
    
    # Check for file
    if "file" not in request.files:
        return jsonify({"ok": False, "error": "No file uploaded"}), 400
    
    file = request.files["file"]
    if not file.filename:
        return jsonify({"ok": False, "error": "Empty filename"}), 400
    
    if not _allowed_file(file.filename):
        return jsonify({
            "ok": False,
            "error": f"Invalid file type. Allowed: {', '.join(ALLOWED_EXTENSIONS)}",
        }), 400
    
    # Get parameters
    try:
        frame_skip = int(request.form.get("frame_skip", "3"))
        frame_skip = max(1, min(30, frame_skip))  # Clamp 1-30
    except ValueError:
        frame_skip = 3
    
    try:
        confidence = float(request.form.get("confidence", "0.4"))
        confidence = max(0.1, min(1.0, confidence))
    except ValueError:
        confidence = 0.4
    
    try:
        overlap = float(request.form.get("overlap", "0.3"))
        overlap = max(0.1, min(1.0, overlap))
    except ValueError:
        overlap = 0.3
    
    # Generate job ID and save video
    job_id = uuid.uuid4().hex[:12]
    job_dir = _get_job_dir(job_id)
    os.makedirs(job_dir, exist_ok=True)
    
    safe_name = _safe_filename(file.filename)
    video_path = os.path.join(job_dir, f"input_{safe_name}")
    file.save(video_path)
    
    # Check file size
    file_size = os.path.getsize(video_path)
    if file_size > MAX_FILE_SIZE:
        os.remove(video_path)
        return jsonify({
            "ok": False,
            "error": f"File too large. Maximum: {MAX_FILE_SIZE // (1024*1024)}MB",
        }), 400
    
    # Get video info for estimation
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.release()
    
    estimated_calls = client.estimate_api_calls(total_frames, frame_skip)
    duration = total_frames / fps if fps > 0 else 0
    
    # Initialize job state
    job_meta = {
        "job_id": job_id,
        "state": "running",
        "progress": 0.0,
        "message": "Starting...",
        "error": None,
        "filename": file.filename,
        "video_path": video_path,
        "frame_skip": frame_skip,
        "confidence": confidence,
        "overlap": overlap,
        "total_frames": total_frames,
        "fps": fps,
        "duration": duration,
        "estimated_api_calls": estimated_calls,
        "created_at": time.time(),
    }
    
    ACTIVE_JOBS[job_id] = job_meta
    _save_job_meta(job_id, job_meta)
    
    # Start background processing
    thread = threading.Thread(
        target=_process_video_job,
        args=(job_id, video_path, frame_skip, confidence, overlap),
        daemon=True,
    )
    thread.start()
    
    return jsonify({
        "ok": True,
        "job_id": job_id,
        "estimated_api_calls": estimated_calls,
        "total_frames": total_frames,
        "duration": duration,
    })


@roboflow_bp.route("/job/<job_id>")
def roboflow_job_viewer(job_id: str):
    """Job viewer page with frame playback."""
    # Try active jobs first, then disk
    meta = ACTIVE_JOBS.get(job_id) or _load_job_meta(job_id)
    
    if not meta:
        abort(404, description="Job not found")
    
    return render_template("roboflow/viewer.html", job_id=job_id, meta=meta)


@roboflow_bp.route("/status/<job_id>")
def roboflow_job_status(job_id: str):
    """Get job status (JSON)."""
    # Try active jobs first, then disk
    meta = ACTIVE_JOBS.get(job_id) or _load_job_meta(job_id)
    
    if not meta:
        return jsonify({"ok": False, "error": "Job not found"}), 404
    
    # List available frames
    job_dir = _get_job_dir(job_id)
    frames_dir = os.path.join(job_dir, "frames")
    available_frames = []
    
    if os.path.isdir(frames_dir):
        for fname in sorted(os.listdir(frames_dir)):
            if fname.endswith(".jpg"):
                # Extract frame index from filename
                try:
                    idx = int(fname.replace("frame_", "").replace(".jpg", ""))
                    available_frames.append(idx)
                except ValueError:
                    pass
    
    return jsonify({
        "ok": True,
        "job_id": job_id,
        "state": meta.get("state", "unknown"),
        "progress": meta.get("progress", 0),
        "message": meta.get("message", ""),
        "error": meta.get("error"),
        "total_frames": meta.get("total_frames", 0),
        "processed_frames": meta.get("processed_frames", 0),
        "current_frame": meta.get("current_frame", 0),
        "api_calls_made": meta.get("api_calls_made", 0),
        "estimated_api_calls": meta.get("estimated_api_calls", 0),
        "frame_skip": meta.get("frame_skip", 1),
        "available_frames": available_frames,
        "fps": meta.get("fps", 30),
    })


@roboflow_bp.route("/frame/<job_id>/<int:frame_idx>")
def roboflow_get_frame(job_id: str, frame_idx: int):
    """Serve an annotated frame image."""
    job_dir = _get_job_dir(job_id)
    frame_path = os.path.join(job_dir, "frames", f"frame_{frame_idx:06d}.jpg")
    
    if not os.path.exists(frame_path):
        abort(404, description="Frame not found")
    
    return send_file(frame_path, mimetype="image/jpeg")


@roboflow_bp.route("/detections/<job_id>")
def roboflow_get_detections(job_id: str):
    """Get all detections for a job (JSON)."""
    job_dir = _get_job_dir(job_id)
    detections_path = os.path.join(job_dir, "detections.jsonl")
    
    if not os.path.exists(detections_path):
        return jsonify({"ok": False, "error": "Detections not found"}), 404
    
    detections = []
    try:
        with open(detections_path, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    detections.append(json.loads(line))
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500
    
    return jsonify({"ok": True, "detections": detections})


@roboflow_bp.route("/download/<job_id>/<file_type>")
def roboflow_download(job_id: str, file_type: str):
    """Download job results (CSV or JSON)."""
    job_dir = _get_job_dir(job_id)
    
    if file_type == "csv":
        path = os.path.join(job_dir, "detections.csv")
        mime = "text/csv"
        download_name = f"roboflow_detections_{job_id}.csv"
    elif file_type == "json":
        path = os.path.join(job_dir, "summary.json")
        mime = "application/json"
        download_name = f"roboflow_summary_{job_id}.json"
    elif file_type == "detections":
        path = os.path.join(job_dir, "detections.jsonl")
        mime = "application/jsonl"
        download_name = f"roboflow_detections_{job_id}.jsonl"
    else:
        abort(400, description="Invalid file type")
    
    if not os.path.exists(path):
        abort(404, description="File not found")
    
    return send_file(path, mimetype=mime, as_attachment=True, download_name=download_name)
