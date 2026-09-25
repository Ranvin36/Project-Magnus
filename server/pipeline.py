"""
Run the pipeline on every video in trajectories/.

Stage 1 (detection) is wired up; physics modelling and spin estimation will
consume DetectionResult.to_arrays() once they exist.

Usage:
  python server/pipeline.py
  python server/pipeline.py --input trajectories --save-video
"""
import argparse
import json
import os
import time

import torch

import detection

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repo root
DEFAULT_INPUT = os.path.join(ROOT, "trajectories")
DEFAULT_WEIGHTS = os.path.join(ROOT, "checkpoints", "cricket_synth_70_15_15_best_model.pt")
DEFAULT_OUTPUT = os.path.join(ROOT, "output", "pipeline")
VIDEO_EXTS = (".mp4", ".mov", ".avi", ".mkv")


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

        if args.save_video:
            video_path = os.path.join(args.output, f"{stem}_tracked.mp4")
            detection.write_annotated_video(result, video_path)
            print(f"  annotated video -> {video_path}")


if __name__ == "__main__":
    main()
