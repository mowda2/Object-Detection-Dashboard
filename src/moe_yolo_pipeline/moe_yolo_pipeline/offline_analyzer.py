import os, time, csv
from typing import Dict, Iterable
import numpy as np
import cv2
from ultralytics import YOLO
import supervision as sv

def _center_xyxy(xyxy):
    x1, y1, x2, y2 = xyxy
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

def _normalize_names(names: Iterable[str]) -> set:
    """
    Normalize requested class names (aliases -> canonical coco names).
    """
    alias = {
        "people": "person",
        "pedestrian": "person", "pedestrians": "person",
        "bike": "bicycle", "bikes": "bicycle", "cycle": "bicycle",
        "motorbike": "motorcycle", "motorbikes": "motorcycle",
        "van": "truck"  # coarse mapping; many models don't have separate 'van'
    }
    out = set()
    for n in names:
        n = (n or "").strip().lower()
        if not n: continue
        out.add(alias.get(n, n))
    return out

def _resolve_allowed_ids(model_names, include_names) -> set:
    """
    Map class names -> IDs based on model.names (dict or list).
    Fallback to common COCO labels if not present.
    """
    # build inverse map from model
    if isinstance(model_names, dict):
        inv = {v.lower(): int(k) for k, v in model_names.items()}
    else:
        inv = {str(v).lower(): i for i, v in enumerate(model_names)}
    ids = set()
    for n in include_names:
        if n in inv:
            ids.add(inv[n])
    # If nothing resolved and it looks like a COCO model, add a sensible default
    if not ids:
        coco_guess = {
            "person": 0, "bicycle": 1, "car": 2, "motorcycle": 3,
            "bus": 5, "truck": 7
        }
        for n in include_names:
            if n in coco_guess:
                ids.add(coco_guess[n])
    return ids

def run_offline_speed_job(job_rec: Dict):
    """
    Expects in job_rec:
      src, out_video, out_csv
    Optional:
      model_path (default: yolo11n.pt), conf (0.25), meters_per_pixel (0.05),
      device ('cuda'|'cpu'), include (comma string or list of names),
      progress_cb (callable), message_cb (callable)
    """
    progress_cb = job_rec.get("progress_cb", lambda p: None)
    message_cb = job_rec.get("message_cb", lambda m: None)

    try:
        src = job_rec["src"]
        out_video = job_rec["out_video"]
        out_csv = job_rec["out_csv"]
        model_path = job_rec.get("model_path", "yolo11n.pt")
        conf = float(job_rec.get("conf", 0.25))
        meters_per_pixel = float(job_rec.get("meters_per_pixel", 0.05))  # 5 cm/px example
        device = job_rec.get("device", "cuda")

        # Include set: default to vehicles
        include = job_rec.get("include", "")
        if isinstance(include, str):
            include_names = [s.strip() for s in include.split(",") if s.strip()]
        else:
            include_names = list(include or [])
        if not include_names:
            include_names = ["car", "motorcycle", "bus", "truck"]  # default: vehicles
        include_names = _normalize_names(include_names)

        message_cb("Opening video…")
        cap = cv2.VideoCapture(src)
        if not cap.isOpened():
            raise RuntimeError("Failed to open input video")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

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

        # Resolve IDs *after* model is loaded (so names are known)
        allowed_ids = _resolve_allowed_ids(model.names, include_names)

        tracker = sv.ByteTrack()
        box_anno = sv.BoxAnnotator()
        label_anno = sv.LabelAnnotator(text_position=sv.Position.TOP_LEFT)

        last_pt: Dict[int, tuple] = {}
        ema_speed: Dict[int, float] = {}

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

                # ---- inference
                y = model(frame, conf=conf, verbose=False)[0]
                det = sv.Detections.from_ultralytics(y)

                # ---- Filter to allowed classes (NumPy-safe)
                if len(det) > 0:
                    if det.class_id is None or not len(det.class_id):
                        det = det[:0]
                    else:
                        keep = np.isin(det.class_id, list(allowed_ids))
                        det = det[keep]

                # ---- tracking
                det = tracker.update_with_detections(det)

                # ---- annotate + log speeds
                labels = []
                now_t = frame_idx / fps
                for i in range(len(det)):
                    tid = int(det.tracker_id[i]) if det.tracker_id is not None else -1
                    cls_id = int(det.class_id[i]) if det.class_id is not None else -1
                    cname = str(model.names.get(cls_id, cls_id)) if isinstance(model.names, dict) else str(cls_id)

                    cx, cy = _center_xyxy(det.xyxy[i])

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

                    labels.append(f"{cname} • ID {tid} • {spd_kmh:0.1f} km/h")
                    writer_csv.writerow([frame_idx, tid, cname, f"{cx:.2f}", f"{cy:.2f}", f"{spd_kmh:.3f}"])

                frame = box_anno.annotate(frame, det)
                frame = label_anno.annotate(frame, det, labels=labels)
                writer.write(frame)

                if total_frames:
                    progress_cb(min(0.99, frame_idx / max(1, total_frames)))
                else:
                    progress_cb(min(0.99, (frame_idx % 2000) / 2000.0))

            message_cb("Finalizing…")

        cap.release()
        writer.release()
        progress_cb(1.0)
        message_cb("Done.")

    except Exception as e:
        job_rec["error"] = str(e)
        progress_cb(-1.0)
        message_cb(f"Error: {e}")
