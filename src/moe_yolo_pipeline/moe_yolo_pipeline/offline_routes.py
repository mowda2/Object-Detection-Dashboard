import os, uuid, threading
from flask import Blueprint, render_template, request, jsonify, send_file
from .offline_analyzer import run_offline_speed_job

offline_bp = Blueprint("offline", __name__, template_folder="templates")

PKG_DIR = os.path.dirname(__file__)
JOBS_DIR = os.path.join(PKG_DIR, "jobs")
os.makedirs(JOBS_DIR, exist_ok=True)

OFFLINE_JOBS = {}   # jid -> dict {state, progress, error, message, src, video, csv, json}

@offline_bp.route("/offline")
def offline_page():
    return render_template("offline.html")

@offline_bp.route("/offline/analyze", methods=["POST"])
def offline_analyze():
    if "file" not in request.files:
        return jsonify({"ok": False, "error": "No file"}), 400

    f = request.files["file"]
    model = request.form.get("model", request.form.get("model_path", "yolo11n.pt"))
    conf = float(request.form.get("conf", "0.25"))
    mpp = float(request.form.get("meters_per_pixel", "0.05"))
    device = request.form.get("device", "cpu")
    include = request.form.get("include", "")   # comma-separated names

    jid = uuid.uuid4().hex[:12]
    job_dir = os.path.join(JOBS_DIR, jid)
    os.makedirs(job_dir, exist_ok=True)
    src_path = os.path.join(job_dir, "input.mp4")
    out_video = os.path.join(job_dir, "overlay.mp4")
    out_csv   = os.path.join(job_dir, "tracks.csv")
    out_json  = os.path.join(job_dir, "summary.json")
    f.save(src_path)

    OFFLINE_JOBS[jid] = {
        "state": "running",
        "progress": 0.0,
        "error": None,
        "message": "Starting…",
        "src": src_path,
        "video": out_video,
        "csv": out_csv,
        "json": out_json,
    }

    def progress_cb(p):
        if p < 0:
            OFFLINE_JOBS[jid]["state"] = "error"
        else:
            OFFLINE_JOBS[jid]["progress"] = float(p)

    def message_cb(msg):
        OFFLINE_JOBS[jid]["message"] = str(msg)

    def _worker():
        rec = {
            "src": src_path,
            "out_video": out_video,
            "out_csv": out_csv,
            "out_json": out_json,        # <-- NEW: analyzer will write stats JSON here
            "model_path": model,
            "conf": conf,
            "meters_per_pixel": mpp,
            "device": device,
            "include": include,
            "progress_cb": progress_cb,
            "message_cb": message_cb,
        }
        run_offline_speed_job(rec)
        if OFFLINE_JOBS[jid]["state"] != "error":
            if rec.get("error"):
                OFFLINE_JOBS[jid]["state"] = "error"
                OFFLINE_JOBS[jid]["error"] = rec["error"]
            else:
                OFFLINE_JOBS[jid]["state"] = "done"

    threading.Thread(target=_worker, daemon=True).start()
    return jsonify({"ok": True, "job_id": jid})

@offline_bp.route("/offline/status/<jid>")
def offline_status(jid):
    job = OFFLINE_JOBS.get(jid)
    if not job:
        return jsonify({"ok": False, "state": "missing", "progress": 0.0}), 404
    done = (job["state"] == "done")
    resp = {
        "ok": True,
        "state": job["state"],
        "progress": job.get("progress", 0.0),
        "message": job.get("message"),
        "video_url": f"/offline/result/{jid}/video" if done else None,
        "csv_url":   f"/offline/result/{jid}/csv"   if done else None,
        "json_url":  f"/offline/result/{jid}/json"  if done else None,
        "error": job.get("error"),
    }
    return jsonify(resp)

@offline_bp.route("/offline/result/<jid>/<kind>")
def offline_result(jid, kind):
    job = OFFLINE_JOBS.get(jid)
    if not job or job["state"] != "done":
        return jsonify({"ok": False}), 404
    if kind == "video":
        return send_file(job["video"], mimetype="video/mp4", as_attachment=False)
    if kind == "csv":
        return send_file(job["csv"], mimetype="text/csv", as_attachment=True, download_name="tracks.csv")
    if kind == "json":
        # If JSON wasn't created (older jobs), fallback to a tiny summary
        if not os.path.exists(job["json"]):
            try:
                import csv, json as _json
                counts = 0
                with open(job["csv"]) as f:
                    r = csv.DictReader(f)
                    for _ in r: counts += 1
                with open(job["json"], "w") as out:
                    _json.dump({"rows": counts}, out, indent=2)
            except Exception:
                pass
        return send_file(job["json"], mimetype="application/json", as_attachment=True, download_name="summary.json")
    return jsonify({"ok": False, "error": "unknown kind"}), 400
