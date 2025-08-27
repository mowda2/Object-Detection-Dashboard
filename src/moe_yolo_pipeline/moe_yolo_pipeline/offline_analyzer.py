import os, time, csv
from typing import Dict
import numpy as np
import cv2
from ultralytics import YOLO
import supervision as sv

def _center_xyxy(xyxy):
    x1, y1, x2, y2 = xyxy
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

def run_offline_speed_job(job_rec: Dict):
    """
    Expects in job_rec:
      src, out_video, out_csv
    Optional:
      model_path (default: yolo11n.pt), conf (0.25), meters_per_pixel (0.05),
      device ('cuda'|'cpu'), progress_cb (callable), message_cb (callable)
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
            # Older Ultralytics versions may not support .to(); inference still works.
            pass

        tracker = sv.ByteTrack()
        box_anno = sv.BoxAnnotator()
        label_anno = sv.LabelAnnotator(text_position=sv.Position.TOP_LEFT)

        last_pt: Dict[int, tuple] = {}
        ema_speed: Dict[int, float] = {}

        with open(out_csv, "w", newline="") as fcsv:
            writer_csv = csv.writer(fcsv)
            writer_csv.writerow(["frame","track_id","cx","cy","speed_kmh"])

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

                # ---- SAFE vehicle filter (no ambiguous truth on NumPy arrays)
                VEHICLE_IDS = (2, 3, 5, 7)  # car, motorcycle, bus, truck (COCO)
                if len(det) > 0:
                    if det.class_id is None:
                        keep = np.zeros(len(det), dtype=bool)
                    else:
                        keep = np.isin(det.class_id, VEHICLE_IDS)
                    det = det[keep]  # empty if nothing matches

                # ---- tracking
                det = tracker.update_with_detections(det)

                # ---- annotate + log speeds
                labels = []
                now_t = frame_idx / fps
                for i in range(len(det)):
                    # tracker_id may be np.ndarray; cast safely
                    tid = int(det.tracker_id[i]) if det.tracker_id is not None else -1
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

                    labels.append(f"ID {tid} • {spd_kmh:0.1f} km/h")
                    writer_csv.writerow([frame_idx, tid, f"{cx:.2f}", f"{cy:.2f}", f"{spd_kmh:.3f}"])

                frame = box_anno.annotate(frame, det)
                frame = label_anno.annotate(frame, det, labels=labels)
                writer.write(frame)

                if total_frames:
                    progress_cb(min(0.99, frame_idx / total_frames))
                else:
                    # unknown length; tick gently
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
