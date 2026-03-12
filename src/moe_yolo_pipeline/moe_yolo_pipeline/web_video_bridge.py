#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, time, threading, uuid, mimetypes, json
from typing import Dict, Optional, List, Tuple, Any

import cv2
from flask import (
    Flask, request, Response, render_template, jsonify,
    send_from_directory, send_file, abort
)

# Import offline analysis blueprint
from .offline_routes import offline_bp
# Import Speed (calibrated) blueprint
from .speed_routes import speed_bp

# ---------- Flask app ----------
HERE = os.path.dirname(__file__)
app = Flask(
    __name__,
    template_folder=os.path.join(HERE, "templates"),
    static_folder=os.path.join(HERE, "static"),
)
app.config["MAX_CONTENT_LENGTH"] = 2 * 1024 * 1024 * 1024  # 2GB
# register offline analysis routes
app.register_blueprint(offline_bp)
# register Speed (calibrated) routes
app.register_blueprint(speed_bp)

JOBS_DIR = os.path.join(HERE, "jobs")
os.makedirs(JOBS_DIR, exist_ok=True)

# ---------- Helpers ----------

# ---------- Process manager ----------


# ---------- Video writer helpers (browser-friendly encoders) ----------
def make_video_writer(out_dir: str, fps: float, size: Tuple[int, int]):
    w, h = size
    cands: List[Tuple[str, object, str, str]] = []

    # Jetson NVENC H.264 (GStreamer)
    mp4_nv = os.path.join(out_dir, "output_h264.mp4")
    gst_nv = (
        f'appsrc ! videoconvert ! nvvidconv ! '
        f'nvv4l2h264enc insert-sps-pps=true iframeinterval=15 bitrate=4000000 preset-level=1 ! '
        f'h264parse ! qtmux faststart=true ! filesink location="{mp4_nv}" sync=false'
    )
    cands.append(("gst", gst_nv, mp4_nv, "video/mp4"))

    # Software x264 (GStreamer)
    mp4_x264 = os.path.join(out_dir, "output_x264.mp4")
    gst_x264 = (
        f'appsrc ! videoconvert ! '
        f'x264enc tune=zerolatency speed-preset=ultrafast bitrate=2000 key-int-max=15 ! '
        f'h264parse ! mp4mux faststart=true ! filesink location="{mp4_x264}" sync=false'
    )
    cands.append(("gst", gst_x264, mp4_x264, "video/mp4"))

    # OpenCV MP4V (fallback)
    mp4_mp4v = os.path.join(out_dir, "output.mp4")
    cands.append(("fourcc", cv2.VideoWriter_fourcc(*"mp4v"), mp4_mp4v, "video/mp4"))

    # MJPEG AVI (last resort)
    avi_mjpg = os.path.join(out_dir, "output.avi")
    cands.append(("fourcc", cv2.VideoWriter_fourcc(*"MJPG"), avi_mjpg, "video/x-msvideo"))

    for kind, param, fname, mime in cands:
        if kind == "gst":
            writer = cv2.VideoWriter(param, cv2.CAP_GSTREAMER, 0, fps, (w, h), True)
        else:
            writer = cv2.VideoWriter(fname, param, fps, (w, h))
        if writer is not None and writer.isOpened():
            return writer, os.path.basename(fname), mime
    raise RuntimeError("Failed to create any video writer (H.264, MP4V, MJPEG).")

# ---------------- Tracking & counting ----------------
def iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[0] if len(boxB) < 4 else boxB[2])
    yB = min(boxA[3], boxB[1] if len(boxB) < 4 else boxB[3])
    inter = max(0, xB - xA) * max(0, yB - yA)
    if inter <= 0:
        return 0.0
    areaA = max(0, boxA[2] - boxA[0]) * max(0, boxA[3] - boxA[1])
    areaB = max(0, (boxB[2] - boxB[0])) * max(0, (boxB[3] - boxB[1]))
    denom = float(areaA + areaB - inter) or 1.0
    return inter / denom

class Track:
    __slots__ = ("tid","bbox","cls","age","last_side")
    def __init__(self, tid, bbox, cls, last_side=None):
        self.tid = tid
        self.bbox = bbox
        self.cls = cls
        self.age = 0
        self.last_side = last_side

class SortishTracker:
    def __init__(self, iou_thresh=0.3, max_age=15):
        self.iou_thresh = iou_thresh
        self.max_age = max_age
        self.next_id = 1
        self.tracks: List[Track] = []

    def update(self, dets):
        matches = []
        unmatched_tracks = list(range(len(self.tracks)))
        unmatched_dets = list(range(len(dets)))
        if self.tracks and dets:
            ious = [[iou(self.tracks[t].bbox, dets[d]["bbox"]) for d in range(len(dets))]
                    for t in range(len(self.tracks))]
            used_tracks = set()
            used_dets = set()
            while True:
                best = 0.0; bt = -1; bd = -1
                for t in range(len(self.tracks)):
                    if t in used_tracks: continue
                    for d in range(len(dets)):
                        if d in used_dets: continue
                        if ious[t][d] > best:
                            best = ious[t][d]; bt = t; bd = d
                if best < self.iou_thresh:
                    break
                matches.append((bt, bd))
                used_tracks.add(bt); used_dets.add(bd)
            unmatched_tracks = [t for t in range(len(self.tracks)) if t not in used_tracks]
            unmatched_dets = [d for d in range(len(dets)) if d not in used_dets]

        for (ti, di) in matches:
            tr = self.tracks[ti]
            tr.bbox = dets[di]["bbox"]
            tr.cls = dets[di]["cls"]
            tr.age = 0

        alive = []
        for ti in unmatched_tracks:
            tr = self.tracks[ti]
            tr.age += 1
            if tr.age <= self.max_age:
                alive.append(tr)

        new_tracks = []
        for di in unmatched_dets:
            d = dets[di]
            new_tracks.append(Track(self.next_id, d["bbox"], d["cls"]))
            self.next_id += 1

        self.tracks = [self.tracks[ti] for (ti,_di) in matches] + alive + new_tracks
        return self.tracks

def center_of(b):
    return ((b[0]+b[2]) * 0.5, (b[1]+b[3]) * 0.5)

# ---------------- Offline analysis job ----------------
class AnalyzeJob:
    def __init__(self, src_path, src_kind, options):
        self.id = uuid.uuid4().hex[:8]
        self.dir = os.path.join(JOBS_DIR, self.id)
        os.makedirs(self.dir, exist_ok=True)
        self.src_path = src_path
        self.src_kind = src_kind  # "video" or "bag"
        self.options = options or {}
        self.state = "queued"
        self.msg = "Queued"
        self.progress = 0.0
        self.results = {}
        self._stop = threading.Event()

    def stop(self): self._stop.set()

    def run(self):
        try:
            self.state = "running"; self.msg = "Initializing…"
            from ultralytics import YOLO
            # Robust device selection (Jetson-safe)
            try:
                import torch
                cuda_ok = bool(torch.cuda.is_available())
            except Exception:
                cuda_ok = False

            model_path = self.options.get("model_path") or self.options.get("model") or "yolov8n.pt"
            conf = float(self.options.get("conf", 0.35))
            line_mode = self.options.get("line_mode", "horizontal")
            imgsz = int(self.options.get("imgsz", 1280))
            iou_thr = float(self.options.get("iou", 0.6))
            agnostic = str(self.options.get("agnostic_nms", "1")).lower() in ("1","true","yes","y","on")
            device_opt = self.options.get("device", None)  # "0" for GPU, "cpu" to force CPU
            if device_opt in (None, "", "auto"):
                device = "0" if cuda_ok else "cpu"
            else:
                device = device_opt

            model = YOLO(model_path)

            # pack inference kwargs once
            self._infer_kwargs = dict(conf=conf, iou=iou_thr, imgsz=imgsz,
                                      agnostic_nms=agnostic, device=device, verbose=False)

            if self.src_kind == "video":
                self._process_video(model, line_mode)
            else:
                raise RuntimeError("Unknown source type")
            self.state = "done"; self.msg = "Completed"; self.progress = 1.0
        except Exception as e:
            self.state = "error"; self.msg = f"{type(e).__name__}: {e}"

    # ---- shared helpers for counting
    @staticmethod
    def _init_counts():
        return {
            "total": 0,
            "by_class": {},
            "by_direction": {"A->B": 0, "B->A": 0},
            "by_class_dir": {}  # label -> {"A->B": n, "B->A": n}
        }

    @staticmethod
    def _on_cross(counts, cname, last_side, cur_side):
        # Determine direction across the line
        direction = "A->B" if (not last_side and cur_side) else "B->A"
        counts["total"] += 1
        counts["by_direction"][direction] = counts["by_direction"].get(direction, 0) + 1
        counts["by_class"][cname] = counts["by_class"].get(cname, 0) + 1
        if cname not in counts["by_class_dir"]:
            counts["by_class_dir"][cname] = {"A->B": 0, "B->A": 0}
        counts["by_class_dir"][cname][direction] += 1

    # ---- unified detector with a second-pass vehicle rescue
    @staticmethod
    def _extract_dets(result) -> List[Dict]:
        dets = []
        if not hasattr(result, "boxes") or result.boxes is None:
            return dets
        for b in result.boxes:
            x1, y1, x2, y2 = map(int, b.xyxy[0].tolist())
            cls = int(b.cls[0])
            dets.append({"bbox": [x1, y1, x2, y2], "cls": cls})
        return dets

    def _detect(self, model, frame):
        # First pass (general) with CUDA→CPU fallback if needed
        try:
            res = model.predict(frame, **self._infer_kwargs)[0]
        except Exception as e:
            msg = str(e)
            if "CUDA" in msg or "cublas" in msg or "cudnn" in msg or "device-side assert" in msg:
                # Switch to CPU and retry once
                self._infer_kwargs["device"] = "cpu"
                res = model.predict(frame, **self._infer_kwargs)[0]
            else:
                raise
        dets = self._extract_dets(res)

        # If we saw zero vehicles, try a quick, lower-conf rescue for vehicle classes
        vehicle_ids = {1, 2, 3, 5, 7}  # bicycle, car, motorcycle, bus, truck
        if not any(d["cls"] in vehicle_ids for d in dets):
            rescue_kwargs = dict(self._infer_kwargs)
            rescue_kwargs["conf"] = max(0.18, float(self._infer_kwargs.get("conf", 0.35)) * 0.7)
            rescue_kwargs["classes"] = sorted(vehicle_ids)
            res2 = model.predict(frame, **rescue_kwargs)[0]
            dets2 = self._extract_dets(res2)
            # merge, preferring first pass for duplicates (by IoU)
            for d2 in dets2:
                if any(iou(d2["bbox"], d1["bbox"]) > 0.5 and d2["cls"] == d1["cls"] for d1 in dets):
                    continue
                dets.append(d2)
        return dets

    def _process_video(self, model, line_mode):
        cap = cv2.VideoCapture(self.src_path)
        if not cap.isOpened():
            raise RuntimeError("Cannot open video")
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(cap.get(cv2.CAP_PROP_FPS)) or 15.0
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0

        writer, video_name, video_mime = make_video_writer(self.dir, fps, (w, h))

        frames_dir = os.path.join(self.dir, "frames")
        os.makedirs(frames_dir, exist_ok=True)
        mjpeg_stride = max(1, int(max(1.0, fps) // 10))

        tracker = SortishTracker(iou_thresh=0.3, max_age=15)
        counts = self._init_counts()
        per_frame_csv = open(os.path.join(self.dir, "counts_per_frame.csv"), "w")

        class_headers: List[str] = []
        header_done = False
        frame_idx = 0
        line_pos = (h // 2 if line_mode == "horizontal" else w // 2)

        def side_of(pt):
            return (pt[1] < line_pos) if line_mode == "horizontal" else (pt[0] < line_pos)

        names = model.names

        while not self._stop.is_set():
            ok, frame = cap.read()
            if not ok:
                break
            frame_idx += 1

            dets = self._detect(model, frame)
            tracks = tracker.update(dets)

            if not header_done:
                class_headers = sorted(set([names[d["cls"]] for d in dets]))
                header = "frame,total" + ("," + ",".join(class_headers) if class_headers else "")
                per_frame_csv.write(header + "\n")
                header_done = True

            per_frame_class = {k: 0 for k in class_headers}

            for tr in tracks:
                cx, cy = center_of(tr.bbox)
                cur_side = side_of((cx, cy))
                if tr.last_side is None:
                    tr.last_side = cur_side
                elif cur_side != tr.last_side:
                    cname = names.get(tr.cls, str(tr.cls))
                    self._on_cross(counts, cname, tr.last_side, cur_side)
                    tr.last_side = cur_side
                    if cname in per_frame_class:
                        per_frame_class[cname] += 1

                cv2.rectangle(frame,(int(tr.bbox[0]),int(tr.bbox[1])),(int(tr.bbox[2]),int(tr.bbox[3])),(0,255,0),2)
                cv2.putText(frame,f"ID{tr.tid}:{names.get(tr.cls,tr.cls)}",
                            (int(tr.bbox[0]),int(tr.bbox[1]-6)),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,0),2)

            if line_mode == "horizontal":
                cv2.line(frame, (0, line_pos), (w, line_pos), (255, 255, 0), 2)
            else:
                cv2.line(frame, (line_pos, 0), (line_pos, h), (255, 255, 0), 2)
            cv2.putText(frame, f"Count: {counts['total']}", (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (50, 220, 255), 2)

            writer.write(frame)

            if frame_idx % mjpeg_stride == 0:
                cv2.imwrite(os.path.join(frames_dir, f"frame_{frame_idx:06d}.jpg"),
                            frame, [int(cv2.IMWRITE_JPEG_QUALITY), 75])

            row = [str(frame_idx), str(counts["total"])]
            for k in class_headers:
                row.append(str(per_frame_class.get(k, 0)))
            per_frame_csv.write(",".join(row) + "\n")

            if total:
                self.progress = min(0.99, frame_idx / max(1, total))
                self.msg = f"Processing frame {frame_idx}/{total}"
            else:
                self.progress = self.progress + 0.002 if self.progress < 0.95 else self.progress
                self.msg = f"Processing frame {frame_idx}"

        per_frame_csv.close()
        writer.release()
        cap.release()

        with open(os.path.join(self.dir, "counts_summary.json"), "w") as f:
            json.dump(counts, f, indent=2)

        self.results = {
            "video": video_name,
            "video_mime": video_mime,
            "csv": "counts_per_frame.csv",
            "summary": "counts_summary.json",
        }

class JobManager:
    def __init__(self):
        self.jobs: Dict[str, AnalyzeJob] = {}
        self.lock = threading.Lock()

    def start(self, src_path, src_kind, options):
        job = AnalyzeJob(src_path, src_kind, options)
        with self.lock:
            self.jobs[job.id] = job
        th = threading.Thread(target=job.run, daemon=True)
        th.start()
        return job

    def get(self, jid):
        with self.lock:
            return self.jobs.get(jid)

    def cancel(self, jid):
        job = self.get(jid)
        if job and job.state in ("queued","running"):
            job.stop()
            job.state = "canceled"
            job.msg = "Canceled by user."
            return True
        return False

JOBS = JobManager()

# ---------- Routes ----------
@app.route("/")
def control_panel():
    return render_template("index.html")

# Serve annotated video & MJPEG fallback
@app.route("/jobs/<job_id>/video")
def job_video(job_id: str):
    job = JOBS.get(job_id)
    mime = None
    if job and job.results and "video" in job.results:
        fname = job.results["video"]
        mime = job.results.get("video_mime")
        path = os.path.join(JOBS_DIR, job_id, fname)
        if os.path.exists(path):
            return send_file(path, mimetype=mime or "video/mp4", as_attachment=False)
    candidates = ["output_h264.mp4", "output_x264.mp4", "output.mp4", "output.avi", "annotated.mp4", "out.mp4"]
    for name in candidates:
        path = os.path.join(JOBS_DIR, job_id, name)
        if os.path.exists(path):
            guessed_mime = "video/mp4" if name.endswith(".mp4") else "video/x-msvideo"
            return send_file(path, mimetype=guessed_mime, as_attachment=False)
    return abort(404, description="Annotated video not found")

@app.route("/jobs/<job_id>/preview.mjpeg")
def job_preview_mjpeg(job_id: str):
    frames_dir = os.path.join(JOBS_DIR, job_id, "frames")
    if not os.path.isdir(frames_dir):
        return abort(404, description="Preview frames not found")

    def gen():
        last_i = -1
        try:
            while True:
                files = sorted(
                    f for f in os.listdir(frames_dir)
                    if f.lower().endswith((".jpg", ".jpeg"))
                )
                if not files:
                    time.sleep(0.1)
                    continue
                if last_i < len(files) - 1:
                    last_i += 1
                fname = os.path.join(frames_dir, files[last_i])
                with open(fname, "rb") as fh:
                    chunk = fh.read()
                yield (b"--frame\r\n"
                       b"Content-Type: image/jpeg\r\n\r\n" + chunk + b"\r\n")
        except (GeneratorExit, BrokenPipeError):
            return
        except Exception:
            time.sleep(0.05)

    return Response(gen(), mimetype="multipart/x-mixed-replace; boundary=frame")

# -------- Offline Analysis UI & APIs (built-in) --------
@app.route("/analyze")
def analyze_page():
    return render_template("analyze.html")

@app.route("/api/analyze/start", methods=["POST"])
def api_analyze_start():
    if "file" not in request.files:
        return jsonify({"ok": False, "error": "No file uploaded"}), 400
    f = request.files["file"]
    if not f.filename:
        return jsonify({"ok": False, "error": "Empty filename"}), 400
    filename = f.filename
    ext = os.path.splitext(filename)[1].lower()

    if ext in (".mp4",".mov",".avi",".mkv",".m4v",".webm"):
        kind = "video"
    else:
        mt = mimetypes.guess_type(filename)[0] or ""
        if "video" in mt:
            kind = "video"
        else:
            return jsonify({"ok": False, "error": f"Unsupported file type: {ext}"}), 400

    tmp_id = uuid.uuid4().hex[:8]
    job_dir = os.path.join(JOBS_DIR, tmp_id)
    os.makedirs(job_dir, exist_ok=True)
    dst = os.path.join(job_dir, filename)
    f.save(dst)

    opts = {
        "model_path": request.form.get("model_path", "yolov8n.pt"),
        "conf": request.form.get("conf", 0.35),
        "line_mode": request.form.get("line_mode", "horizontal"),
        "imgsz": request.form.get("imgsz", 1280),
        "iou": request.form.get("iou", 0.6),
        "agnostic_nms": request.form.get("agnostic_nms", "1"),
        "device": request.form.get("device", None),
    }

    job = JOBS.start(dst, kind, opts)
    return jsonify({"ok": True, "job_id": job.id})

@app.route("/api/analyze/status/<jid>")
def api_analyze_status(jid):
    job = JOBS.get(jid)
    if not job:
        return jsonify({"ok": False, "error": "No such job"}), 404
    return jsonify({
        "ok": True,
        "job_id": job.id,
        "state": job.state,
        "message": job.msg,
        "progress": round(float(job.progress) * 100, 1),
        "results": job.results
    })

@app.route("/api/analyze/cancel/<jid>", methods=["POST"])
def api_analyze_cancel(jid):
    ok = JOBS.cancel(jid)
    return jsonify({"ok": ok})

@app.route("/results/<jid>/<path:name>")
def results_file(jid, name):
    job = JOBS.get(jid)
    if not job:
        return "Not found", 404
    return send_from_directory(job.dir, name, as_attachment=False)

# ---- Aliases to match analyze.html (POST /api/analyze, GET /api/analyze/status?job=ID)
@app.route("/api/analyze", methods=["POST"])
def api_analyze_alias():
    if "file" not in request.files:
        return jsonify({"ok": False, "error": "No file uploaded"}), 400
    f = request.files["file"]
    if not f.filename:
        return jsonify({"ok": False, "error": "Empty filename"}), 400

    filename = f.filename
    ext = os.path.splitext(filename)[1].lower()
    if ext in (".mp4",".mov",".avi",".mkv",".m4v",".webm"):
        kind = "video"
    else:
        mt = mimetypes.guess_type(filename)[0] or ""
        if "video" in mt:
            kind = "video"
        else:
            return jsonify({"ok": False, "error": "Unsupported file type: %s" % ext}), 400

    tmp_id = uuid.uuid4().hex[:8]
    job_dir = os.path.join(JOBS_DIR, tmp_id)
    os.makedirs(job_dir, exist_ok=True)
    dst = os.path.join(job_dir, filename)
    f.save(dst)

    opts = {
        "model_path": request.form.get("model", request.form.get("model_path", "yolov8n.pt")),
        "conf": request.form.get("conf", 0.35),
        "line_mode": request.form.get("line_mode", "horizontal"),
        "imgsz": request.form.get("imgsz", 1280),
        "iou": request.form.get("iou", 0.6),
        "agnostic_nms": request.form.get("agnostic_nms", "1"),
        "device": request.form.get("device", None),
    }
    job = JOBS.start(dst, kind, opts)
    return jsonify({"ok": True, "job_id": job.id})

@app.route("/api/analyze/status")
def api_analyze_status_alias():
    jid = (request.args.get("job") or "").strip()
    job = JOBS.get(jid)
    if not job:
        return jsonify({"ok": False, "error": "No such job"}), 404

    results = job.results or {}
    csv_url  = f"/results/{job.id}/{results['csv']}"     if "csv" in results else None
    json_url = f"/results/{job.id}/{results['summary']}" if "summary" in results else None
    mp4_path = os.path.join(JOBS_DIR, job.id, results.get("video","")) if "video" in results else ""
    mp4_url  = f"/jobs/{job.id}/video" if mp4_path and os.path.exists(mp4_path) else None
    mjpeg_url = f"/jobs/{job.id}/preview.mjpeg"

    return jsonify({
        "ok": True,
        "job_id": job.id,
        "state": job.state,
        "message": job.msg,
        "progress": round(float(job.progress or 0.0) * 100, 1),
        "csv": csv_url,
        "json": json_url,
        "video": mp4_url,
        "mjpeg": mjpeg_url,
    })

# ---- Summary endpoint (unchanged)
@app.route("/api/analyze/summary")
def api_analyze_summary():
    jid = (request.args.get("job") or "").strip()
    job = JOBS.get(jid)
    if not job:
        return jsonify({"ok": False, "error": "No such job"}), 404

    path = os.path.join(JOBS_DIR, job.id, "counts_summary.json")
    if not os.path.exists(path):
        return jsonify({"ok": False, "error": "Summary not found yet"}), 404

    with open(path, "r") as f:
        data = json.load(f)

    by_class = data.get("by_class", {})
    response = {
        "ok": True,
        "job_id": job.id,
        "total": int(data.get("total", 0)),
        "categories": {
            "pedestrians": int(by_class.get("person", 0)),
            "bicycles":   int(by_class.get("bicycle", 0)),
            "cars":       int(by_class.get("car", 0)),
        },
        "by_class": {k: int(v) for k, v in by_class.items()},
        "by_direction": {k: int(v) for k, v in data.get("by_direction", {}).items()},
        "by_class_dir": {k: {"A->B": int(v.get("A->B",0)), "B->A": int(v.get("B->A",0))}
                         for k, v in data.get("by_class_dir", {}).items()},
    }
    return jsonify(response)

if __name__ == "__main__":
    print("[TrafficIQ] Web: http://0.0.0.0:5000")
    app.run(host="0.0.0.0", port=5000, threaded=True)
