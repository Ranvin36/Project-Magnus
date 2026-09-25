"""
Ball detection stage: video -> per-frame ball pixel position (u, v).

Ported from run_tracker_on_video in notebooks/ball_detection_tracking.ipynb, with two
changes: frames are streamed through the model instead of all being held in
memory, and the result also carries fps / frame size, since the physics
stage needs real time stamps (t = frame_index / fps) and the intrinsics are
defined at the original video resolution.
"""
import os
import sys
from dataclasses import dataclass

import cv2
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "training"))
import resume_training as rt  # noqa: E402

RESIZE_TO = (512, 288)  # (W, H) the model was trained at
WINDOW_SIZE = rt.WINDOW_SIZE


@dataclass
class DetectionResult:
    video_path: str
    fps: float
    width: int
    height: int
    # One entry per frame: (u, v, conf) in original-resolution pixels, or
    # None where no ball was accepted.
    detections: list

    def to_arrays(self):
        """Visible frames only, as arrays ready for the physics fit:
        uv (N,2) pixels, t (N,) seconds, conf (N,), frame_idx (N,)."""
        idx = np.array([i for i, d in enumerate(self.detections) if d is not None], dtype=int)
        if len(idx) == 0:
            return np.zeros((0, 2)), np.zeros(0), np.zeros(0), idx
        dets = np.array([self.detections[i] for i in idx], dtype=float)
        return dets[:, :2], idx / self.fps, dets[:, 2], idx


def load_model(weights_path, device):
    model = rt.TrackNetV2().to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device, weights_only=True))
    model.eval()
    return model


def _preprocess(frame):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, RESIZE_TO, interpolation=cv2.INTER_AREA)
    return resized.astype(np.float32).transpose(2, 0, 1) / 255.0


def _raw_predictions(video_path, model, device):
    """Heatmap argmax + confidence for every frame, scaled to original size."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Could not open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    scale_x = width / RESIZE_TO[0]
    scale_y = height / RESIZE_TO[1]

    raw_conf, raw_pos = [], []
    done = False
    with torch.no_grad():
        while not done:
            group = []
            while len(group) < WINDOW_SIZE:
                ok, frame = cap.read()
                if not ok:
                    done = True
                    break
                group.append(_preprocess(frame))
            if not group:
                break
            n_valid = len(group)
            group += [group[-1]] * (WINDOW_SIZE - n_valid)

            tensor = torch.from_numpy(np.concatenate(group, axis=0)).unsqueeze(0).to(device)
            probs = torch.sigmoid(model(tensor))[0].cpu().numpy()  # (3, H, W)

            for k in range(n_valid):
                prob_map = probs[k]
                py, px = np.unravel_index(np.argmax(prob_map), prob_map.shape)
                raw_conf.append(float(prob_map.max()))
                raw_pos.append((px * scale_x, py * scale_y))
    cap.release()
    return raw_conf, raw_pos, fps, width, height


def _persistence_filter(raw_conf, threshold, min_consecutive):
    # Real detections sit at ~0.98-1.00 for many consecutive frames; false
    # spikes are isolated 1-2 frame blips. Drop runs shorter than
    # min_consecutive.
    n = len(raw_conf)
    above = [c >= threshold for c in raw_conf]
    persistent = above[:]
    i = 0
    while i < n:
        if above[i]:
            j = i
            while j < n and above[j]:
                j += 1
            if j - i < min_consecutive:
                for k in range(i, j):
                    persistent[k] = False
            i = j
        else:
            i += 1
    return persistent


def _trajectory_gate(candidates, max_accel_px, high_conf_override, max_gap_frames):
    # Accept a candidate only if it lands near the constant-velocity
    # extrapolation from the last accepted point. Resets after a long gap so
    # the ball can be reacquired anywhere.
    detections = [None] * len(candidates)
    last_pos = last_frame = velocity = None

    for i, cand in enumerate(candidates):
        if cand is None:
            if last_frame is not None and i - last_frame > max_gap_frames:
                last_pos = last_frame = velocity = None
            continue

        x, y, conf = cand
        accept = True
        if last_pos is not None:
            dt = i - last_frame
            if dt > max_gap_frames:
                velocity = None
            if velocity is not None:
                pred_x = last_pos[0] + velocity[0] * dt
                pred_y = last_pos[1] + velocity[1] * dt
                deviation = ((x - pred_x) ** 2 + (y - pred_y) ** 2) ** 0.5
                if deviation > max_accel_px * dt and conf < high_conf_override:
                    accept = False

        if accept:
            detections[i] = (x, y, conf)
            if last_pos is not None:
                dt = i - last_frame
                velocity = ((x - last_pos[0]) / dt, (y - last_pos[1]) / dt)
            last_pos = (x, y)
            last_frame = i
    return detections


def detect(video_path, model, device, threshold=0.9, min_consecutive=2,
           max_accel_px=None, high_conf_override=0.97, max_gap_frames=10):
    raw_conf, raw_pos, fps, width, height = _raw_predictions(video_path, model, device)

    if max_accel_px is None:
        max_accel_px = max(15.0, 0.02 * width)

    persistent = _persistence_filter(raw_conf, threshold, min_consecutive)
    candidates = [
        (raw_pos[i][0], raw_pos[i][1], raw_conf[i]) if persistent[i] else None
        for i in range(len(raw_conf))
    ]
    detections = _trajectory_gate(candidates, max_accel_px, high_conf_override, max_gap_frames)
    return DetectionResult(video_path, fps, width, height, detections)


def write_annotated_video(result, output_path, trail_len=7):
    """Re-read the source video and draw the accepted detections on it."""
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    cap = cv2.VideoCapture(result.video_path)
    writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"),
                             result.fps, (result.width, result.height))
    trail = []
    for det in result.detections:
        ok, frame = cap.read()
        if not ok:
            break
        if det is not None:
            trail.append(det[:2])
            if len(trail) > trail_len:
                trail.pop(0)

        for j, (tx, ty) in enumerate(trail):
            alpha = (j + 1) / len(trail)
            cv2.circle(frame, (int(tx), int(ty)), 3 + int(4 * alpha), (0, int(255 * alpha), 255), -1)

        if det is not None:
            x, y, conf = det
            cv2.circle(frame, (int(x), int(y)), 8, (0, 0, 255), 2)
            cv2.putText(frame, f"{conf:.2f}", (int(x) + 10, int(y) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
        writer.write(frame)
    cap.release()
    writer.release()
