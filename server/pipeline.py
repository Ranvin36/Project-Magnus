"""
Run the pipeline on every video in trajectories/.

Stage 1: detection   video -> ball pixel position per frame (detection.py)
Stage 2: 3D          pixel positions + ball widths -> 3D flight (reconstruct.py)

The 3D stage also needs the camera's focal length and the ball's width in
pixels. The tracker can't measure widths yet, so for now both come from the
cricket-synth label of the clip (matched by file name, e.g.
main1000_000009_test.mp4 -> labels/main1000_000009.json). The ball positions
always come from the tracker. Spin estimation doesn't exist yet.

Usage:
  python server/pipeline.py
  python server/pipeline.py --input trajectories --save-video
"""
import argparse
import json
import os
import re
import time

import numpy as np

import torch

import detection
import reconstruct

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repo root
DEFAULT_INPUT = os.path.join(ROOT, "trajectories")
DEFAULT_WEIGHTS = os.path.join(ROOT, "checkpoints", "cricket_synth_70_15_15_best_model.pt")
DEFAULT_OUTPUT = os.path.join(ROOT, "output", "pipeline")
VIDEO_EXTS = (".mp4", ".mov", ".avi", ".mkv")
LABELS_DIR = r"D:\Git-Repos\MAGNUS\Ball-Tracking\cricket-synth\out\labels"


def detections_to_dict(result):
    frames = [
        {"frame": i, "t": i / result.fps, "u": d[0], "v": d[1], "conf": d[2]}
        for i, d in enumerate(result.detections) if d is not None
    ]
    return {
        "video": os.path.basename(result.video_path),
        "fps": result.fps,
        "width": result.width,
        "height": result.height,
        "n_frames": len(result.detections),
        "detections": frames,
    }


def find_label(video_name):
    """The cricket-synth label for a clip, or None. Matches the clip id in the
    file name, e.g. 'main1000_000009_test.mp4' -> 'main1000_000009.json'."""
    stem = os.path.splitext(os.path.basename(video_name))[0]
    for clip_id in (stem, re.sub(r"_test$", "", stem)):
        path = os.path.join(LABELS_DIR, clip_id + ".json")
        if os.path.isfile(path):
            with open(path) as f:
                return json.load(f)
    return None


def reconstruct_3d(result, label):
    """Stage 2. Tracker positions + label focal length and ball widths -> 3D.
    Raises ValueError with a readable reason when it can't be done."""
    if label is None:
        raise ValueError("No 3D: the tracker can't measure the ball's width yet, and no "
                         "cricket-synth label matches this file name to supply it.")
    cam = label["camera"]["intrinsics"]
    # Label frames count from 1, video frames from 0.
    truth = {f["frame"] - 1: f for f in label["frames"] if f["visible"]}

    # Only detections with a ball width go in. Where the label says the ball
    # isn't in the picture, the tracker's detection is a false one.
    idx = [i for i, d in enumerate(result.detections) if d is not None and i in truth]
    if len(idx) < 10:
        raise ValueError(f"No 3D: only {len(idx)} usable detections (need 10).")
    uv = np.array([result.detections[i][:2] for i in idx])
    width = np.array([2 * truth[i]["radius_px"] for i in idx])
    out = reconstruct.reconstruct(np.array(idx) / result.fps, uv, width,
                                  cam["fx"], cam["cx"], cam["cy"], result.fps)
    out["dropped_detections"] = sum(d is not None for d in result.detections) - len(idx)

    # The true path, in the same frame as the reconstruction (origin = where the
    # ball was first seen, forward = along the pitch), to compare against.
    # cricket-synth: the ball travels along -Y, X across, Z up, pitch at z=0.015.
    first = truth[out["frames"][0]["frame"]]["world_m"]
    out["truth"] = [dict(frame=i, forward=round(first[1] - p[1], 3),
                         right=round(first[0] - p[0], 3), height=round(p[2] - 0.015, 3))
                    for i, f in sorted(truth.items()) for p in [f["world_m"]]]
    out["true_speed_kmh"] = label["delivery"]["speed_kmh"]
    return out


def save_detections(result, path):
    with open(path, "w") as f:
        json.dump(detections_to_dict(result), f, indent=1)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", default=DEFAULT_INPUT, help="directory of videos")
    parser.add_argument("--weights", default=DEFAULT_WEIGHTS)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--threshold", type=float, default=0.9)
    parser.add_argument("--save-video", action="store_true", help="also write annotated videos")
    args = parser.parse_args()

    if not os.path.isdir(args.input):
        raise SystemExit(f"Input directory not found: {args.input}")
    videos = sorted(f for f in os.listdir(args.input) if f.lower().endswith(VIDEO_EXTS))
    if not videos:
        raise SystemExit(f"No videos ({', '.join(VIDEO_EXTS)}) in {args.input}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Weights: {args.weights}")
    model = detection.load_model(args.weights, device)
    os.makedirs(args.output, exist_ok=True)

    for name in videos:
        stem = os.path.splitext(name)[0]
        start = time.time()
        result = detection.detect(os.path.join(args.input, name), model, device, threshold=args.threshold)
        uv, t, conf, _ = result.to_arrays()

        print(f"\n{name}: {len(result.detections)} frames @ {result.fps:.1f} fps "
              f"({result.width}x{result.height}), {time.time() - start:.1f}s")
        print(f"  ball detected in {len(uv)}/{len(result.detections)} frames")
        if len(uv):
            print(f"  first seen t={t[0]:.2f}s at (u,v)=({uv[0, 0]:.0f}, {uv[0, 1]:.0f}), "
                  f"last seen t={t[-1]:.2f}s at ({uv[-1, 0]:.0f}, {uv[-1, 1]:.0f})")

        json_path = os.path.join(args.output, f"{stem}_detections.json")
        save_detections(result, json_path)
        print(f"  detections -> {json_path}")

        try:
            flight = reconstruct_3d(result, find_label(name))
            json_path = os.path.join(args.output, f"{stem}_3d.json")
            with open(json_path, "w") as f:
                json.dump(flight, f, indent=1)
            print(f"  3D: {flight['n_flights']} flights, speed {flight['metrics']['speed_first_seen_kmh']} km/h, "
                  f"fit error {flight['metrics']['fit_error_px']} px -> {json_path}")
        except ValueError as err:
            print(f"  {err}")

        if args.save_video:
            video_path = os.path.join(args.output, f"{stem}_tracked.mp4")
            detection.write_annotated_video(result, video_path)
            print(f"  annotated video -> {video_path}")


if __name__ == "__main__":
    main()
