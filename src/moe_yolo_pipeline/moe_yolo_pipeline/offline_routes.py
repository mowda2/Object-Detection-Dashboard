import os, uuid, threading, hashlib, json, shutil
from typing import Dict
from flask import Blueprint, render_template, request, jsonify, send_file, abort

from .offline_analyzer import run_offline_speed_job
from .library_db import LibraryDB

offline_bp = Blueprint("offline", __name__, template_folder="templates")

PKG_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(PKG_DIR, "data")
MEDIA_DIR = os.path.join(DATA_DIR, "media")
JOBS_DIR  = os.path.join(PKG_DIR, "jobs")   # artifacts per analysis id
DB_PATH   = os.path.join(DATA_DIR, "library.sqlite3")
os.makedirs(MEDIA_DIR, exist_ok=True)
os.makedirs(JOBS_DIR, exist_ok=True)

LIB = LibraryDB(DB_PATH)
OFFLINE_JOBS: Dict[str, Dict] = {}   # in-flight jobs only

APP_VERSION = "1.0"  # include in params hash so logic changes won't collide caches

# ---------- small helpers ----------
def _sha256_file(path, buf=1024*1024):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(buf)
            if not b: break
            h.update(b)
    return h.hexdigest()

def _params_hash(d: Dict):
    keep = {
        "model_path": d.get("model_path"),
        "include": d.get("include"),
        "conf": float(d.get("conf", 0.25)),
        "meters_per_pixel": float(d.get("meters_per_pixel", 0.05)),
        "device": d.get("device", "cpu"),
        "app_version": APP_VERSION,
    }
    s = json.dumps(keep, sort_keys=True, separators=(",",":"))
    return hashlib.sha256(s.encode()).hexdigest()

def _analysis_id(file_hash: str, params_hash: str) -> str:
    return hashlib.sha256((file_hash + params_hash).encode()).hexdigest()[:12]

# ---------- pages ----------
@offline_bp.route("/offline")
def offline_page():
    return render_template("offline.html")

@offline_bp.route("/library")
def library_page():
    return render_template("library.html")

# ---------- REST: library ----------
@offline_bp.route("/api/library")
def api_library_list():
    q = request.args.get("q","")
    items = LIB.list_analyses(q=q, limit=200, offset=0)
    # decorate with URLs
    for it in items:
        it["video_url"]  = f"/offline/result/{it['id']}/video"
        it["csv_url"]    = f"/offline/result/{it['id']}/csv"
        it["json_url"]   = f"/offline/result/{it['id']}/json"
        it["poster_url"] = f"/offline/result/{it['id']}/poster"
    return jsonify({"ok": True, "items": items})

@offline_bp.route("/api/library/<aid>")
def api_library_detail(aid):
    row = LIB.get_analysis(aid)
    if not row: return jsonify({"ok": False, "error": "not found"}), 404
    stats = LIB.list_analyses(q="")  # quick way to get stats; small volume expected
    cs = {}
    for r in stats:
        if r["id"] == aid:
            cs = r.get("class_stats", {})
            break
    return jsonify({
        "ok": True,
        "item": dict(row),
        "class_stats": cs,
        "video_url":  f"/offline/result/{aid}/video",
        "csv_url":    f"/offline/result/{aid}/csv",
        "json_url":   f"/offline/result/{aid}/json",
        "poster_url": f"/offline/result/{aid}/poster",
    })

@offline_bp.route("/api/library/<aid>/delete", methods=["POST", "DELETE"])
def api_library_delete(aid):
    row = LIB.get_analysis(aid)
    if not row: return jsonify({"ok": False, "error": "not found"}), 404
    # delete artifact folder
    job_dir = os.path.join(JOBS_DIR, aid)
    try:
        if os.path.isdir(job_dir):
            shutil.rmtree(job_dir, ignore_errors=True)
    except Exception:
        pass
    LIB.delete_analysis(aid)
    return jsonify({"ok": True})

# ---------- REST: analyze with dedup ----------
@offline_bp.route("/offline/analyze", methods=["POST"])
def offline_analyze():
    if "file" not in request.files:
        return jsonify({"ok": False, "error": "No file"}), 400

    f = request.files["file"]
    filename = f.filename or "upload.bin"
    ext = os.path.splitext(filename)[1].lower()

    # Save to temp then hash
    tmp_id = uuid.uuid4().hex[:8]
    tmp_dir = os.path.join(JOBS_DIR, f"tmp_{tmp_id}")
    os.makedirs(tmp_dir, exist_ok=True)
    tmp_path = os.path.join(tmp_dir, filename)
    f.save(tmp_path)
    size_bytes = os.path.getsize(tmp_path)
    file_hash = _sha256_file(tmp_path)

    # Move (or keep) into media store (dedup by hash)
    media_name = f"{file_hash}{ext if ext else ''}"
    media_path = os.path.join(MEDIA_DIR, media_name)
    if not os.path.exists(media_path):
        os.rename(tmp_path, media_path)
    else:
        os.remove(tmp_path)
    try:
        shutil.rmtree(tmp_dir, ignore_errors=True)
    except Exception:
        pass

    # record/refresh file row
    LIB.upsert_file(file_hash, filename, size_bytes, media_path)

    # Params & dedup key
    include = request.form.get("include", "")
    params = {
        "model_path": request.form.get("model", request.form.get("model_path", "yolo11n.pt")),
        "include": include,
        "conf": float(request.form.get("conf", "0.25")),
        "meters_per_pixel": float(request.form.get("meters_per_pixel", "0.05")),
        "device": request.form.get("device", "cpu"),
    }
    p_hash = _params_hash(params)
    aid = _analysis_id(file_hash, p_hash)

    # Already analyzed?
    found = LIB.find_analysis(file_hash, p_hash)
    if found:
        return jsonify({"ok": True, "job_id": found["id"], "cached": True})

    # New job → define artifact paths
    job_dir = os.path.join(JOBS_DIR, aid)
    os.makedirs(job_dir, exist_ok=True)
    out_video  = os.path.join(job_dir, "overlay.mp4")
    out_csv    = os.path.join(job_dir, "tracks.csv")
    out_json   = os.path.join(job_dir, "summary.json")
    out_poster = os.path.join(job_dir, "poster.jpg")

    OFFLINE_JOBS[aid] = {
        "state": "running",
        "progress": 0.0,
        "error": None,
        "message": "Starting…",
    }

    def progress_cb(p):
        if p < 0:
            OFFLINE_JOBS[aid]["state"] = "error"
        else:
            OFFLINE_JOBS[aid]["progress"] = float(p)

    def message_cb(msg):
        OFFLINE_JOBS[aid]["message"] = str(msg)

    def _worker():
        import logging
        logging.basicConfig(level=logging.DEBUG)
        logger = logging.getLogger(__name__)
        logger.info(f"[_worker] Starting job {aid}")
        
        # run analyzer (reads from media_path)
        rec = {
            "src": media_path,
            "original_filename": filename,  # PASS 2: for summary.json naming
            "out_video": out_video,
            "out_csv": out_csv,
            "out_json": out_json,
            "poster_path": out_poster,
            "model_path": params["model_path"],
            "conf": params["conf"],
            "meters_per_pixel": params["meters_per_pixel"],
            "device": params["device"],
            "include": include,
            "progress_cb": progress_cb,
            "message_cb": message_cb,
        }
        run_offline_speed_job(rec)
        logger.info(f"[_worker] run_offline_speed_job completed for {aid}, state={OFFLINE_JOBS[aid]['state']}")

        if OFFLINE_JOBS[aid]["state"] == "error" or rec.get("error"):
            OFFLINE_JOBS[aid]["state"] = "error"
            OFFLINE_JOBS[aid]["error"] = rec.get("error")
            logger.error(f"[_worker] Job {aid} failed with error: {rec.get('error')}")
            return

        # Insert analysis row into DB from produced artifacts
        fps = frames = None
        class_counts = {}
        try:
            with open(out_json, "r") as jf:
                j = json.load(jf)
                fps = j.get("fps")
                frames = j.get("frames")
                class_counts = j.get("unique_objects_by_class", {}) or {}
            logger.info(f"[_worker] Loaded summary.json: fps={fps}, frames={frames}, classes={len(class_counts)}")
        except Exception as e:
            logger.error(f"[_worker] Failed to load summary.json: {e}")

        logger.info(f"[_worker] Calling LIB.insert_analysis for {aid}")
        LIB.insert_analysis({
            "id": aid,
            "file_hash": file_hash,
            "params_hash": p_hash,
            "model": params["model_path"],
            "include": include,
            "conf": params["conf"],
            "mpp": params["meters_per_pixel"],
            "device": params["device"],
            "fps": fps,
            "frames": frames,
            "video_path": out_video,
            "csv_path": out_csv,
            "json_path": out_json,
            "poster_path": out_poster,
        }, class_counts)
        logger.info(f"[_worker] insert_analysis completed for {aid}")

        OFFLINE_JOBS[aid]["state"] = "done"
        OFFLINE_JOBS[aid]["progress"] = 1.0
        OFFLINE_JOBS[aid]["message"] = "Done."
        logger.info(f"[_worker] Job {aid} completed successfully")

    threading.Thread(target=_worker, daemon=True).start()
    return jsonify({"ok": True, "job_id": aid, "cached": False})

@offline_bp.route("/offline/status/<jid>")
def offline_status(jid):
    j = OFFLINE_JOBS.get(jid)
    if j:
        done = (j.get("state") == "done")
        return jsonify({
            "ok": True,
            "state": j.get("state"),
            "progress": j.get("progress", 0.0),
            "message": j.get("message"),
            "video_url": f"/offline/result/{jid}/video" if done else None,
            "csv_url":   f"/offline/result/{jid}/csv"   if done else None,
            "json_url":  f"/offline/result/{jid}/json"  if done else None,
        })

    # Not an active job? See if it's in the DB (cached result).
    row = LIB.get_analysis(jid)
    if not row:
        return jsonify({"ok": False, "state": "missing", "progress": 0.0}), 404
    return jsonify({
        "ok": True,
        "state": "done",
        "progress": 1.0,
        "message": "Cached result",
        "video_url": f"/offline/result/{jid}/video",
        "csv_url":   f"/offline/result/{jid}/csv",
        "json_url":  f"/offline/result/{jid}/json",
    })

@offline_bp.route("/offline/result/<aid>/<kind>")
def offline_result(aid, kind):
    # Prefer active job artifacts; else from DB
    job = OFFLINE_JOBS.get(aid)
    if job and job.get("state") == "done":
        base = os.path.join(JOBS_DIR, aid)
        mapping = {
            "video":  ("overlay.mp4", "video/mp4"),
            "csv":    ("tracks.csv", "text/csv"),
            "json":   ("summary.json", "application/json"),
            "poster": ("poster.jpg", "image/jpeg"),
        }
        if kind in mapping:
            name, mime = mapping[kind]
            path = os.path.join(base, name)
            if os.path.exists(path):
                return send_file(path, mimetype=mime, as_attachment=False)
        return abort(404)

    row = LIB.get_analysis(aid)
    if not row:
        return abort(404)
    path = None; mime = None
    if kind == "video":  path, mime = row["video_path"], "video/mp4"
    if kind == "csv":    path, mime = row["csv_path"], "text/csv"
    if kind == "json":   path, mime = row["json_path"], "application/json"
    if kind == "poster": path, mime = row["poster_path"], "image/jpeg"
    if not path or not os.path.exists(path):
        return abort(404)
    return send_file(path, mimetype=mime, as_attachment=False)
