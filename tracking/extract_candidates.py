"""
Run a detector over dataset splits and save every frame's candidates.

Output: data/<dataset>/derived/candidates_<tag>_<split>.json
  {"detector": {...}, "split", "splits_file", "clips": {clip_id: {"fps", "native_size", "frames": [
      {"t", "vis", "candidates": [...], "gt": [x, y] or null,
       "visible", "occluded", "hard"}]}}}
All positions are in 512x288 pixels. Each frame is scored once, as the middle of its
3-frame window where possible (first/last frame use the window at the clip edge).

Defaults to the frozen detector (tracking/frozen.py). Out-of-fold detectors:
  python -m tracking.extract_candidates --run tennis_foldA --epoch 12 --vis-threshold 0.6 \
      --splits-file splits_foldA.json test
Usage (frozen detector): python -m tracking.extract_candidates val test
"""
import argparse
import json
import os
import time

import cv2
import numpy as np
import torch

from tracking import frozen
from tracking.candidates import BLOB_THRESHOLD, MAX_CANDIDATES, extract_candidates

DATA = os.path.join(frozen.REPO, "data", frozen.DATASET, "512x288")
OUT_DIR = os.path.join(frozen.REPO, "data", frozen.DATASET, "derived")


def candidates_for_clips(model, clip_ids, device, data_root=DATA):
    """{clip_id: {"fps", "native_size", "frames": [...]}} for the given clips."""
    clips = {}
    with torch.no_grad():
        for cid in clip_ids:
            meta = json.load(open(os.path.join(data_root, "labels", cid + ".json")))
            W, H = meta["camera"]["intrinsics"]["width"], meta["camera"]["intrinsics"]["height"]
            frames = [cv2.cvtColor(cv2.imread(os.path.join(data_root, "clips", cid, meta["frame_pattern"] % e["frame"])), cv2.COLOR_BGR2RGB)
                      for e in meta["frames"]]
            n = len(frames)
            starts = [min(max(t - 1, 0), n - 3) for t in range(n)]
            rows = []
            for b in range(0, n, 4):
                idx = list(range(b, min(b + 4, n)))
                x = torch.stack([torch.from_numpy(np.concatenate([frames[starts[t] + j].transpose(2, 0, 1) for j in range(3)]).astype(np.float32) / 255.0)
                                 for t in idx]).to(device)
                with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                    h, v = model(x)
                h, v = torch.sigmoid(h.float()).cpu().numpy(), torch.sigmoid(v.float()).cpu().numpy()
                for k, t in enumerate(idx):
                    pos = t - starts[t]
                    e = meta["frames"][t]
                    gt = [e["centre_px"][0] * 512 / W, e["centre_px"][1] * 288 / H] if e["centre_px"] else None
                    rows.append({"t": t, "vis": float(v[k, pos]), "candidates": extract_candidates(h[k, pos]),
                                 "gt": gt, "visible": bool(e["visible"]), "occluded": bool(e["occluded"]),
                                 "hard": bool(e.get("hard"))})
            clips[cid] = {"fps": meta["fps"], "native_size": [W, H], "frames": rows}
    return clips


def main():
    p = argparse.ArgumentParser()
    p.add_argument("splits", nargs="*", default=["val", "test"], help="split names inside the split file")
    p.add_argument("--run", default=frozen.RUN)
    p.add_argument("--epoch", type=int, default=frozen.EPOCH)
    p.add_argument("--vis-threshold", type=float, default=frozen.VIS_THRESHOLD)
    p.add_argument("--splits-file", default="splits.json")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = frozen.load_detector(device, run=args.run, epoch=args.epoch)
    tag = f"{args.run}_ep{args.epoch:02d}"
    spec = json.load(open(os.path.join(DATA, args.splits_file)))
    for split in args.splits:
        t0 = time.time()
        out = {"detector": {"run": args.run, "epoch": args.epoch, "vis_threshold": args.vis_threshold,
                            "blob_threshold": BLOB_THRESHOLD, "max_candidates": MAX_CANDIDATES, "coords": "512x288"},
               "split": split, "splits_file": args.splits_file,
               "clips": candidates_for_clips(model, spec[split], device)}
        os.makedirs(OUT_DIR, exist_ok=True)
        path = os.path.join(OUT_DIR, f"candidates_{tag}_{split}.json")
        with open(path, "w") as f:
            json.dump(out, f)
        n_frames = sum(len(c["frames"]) for c in out["clips"].values())
        print(f"{split}: {len(spec[split])} clips, {n_frames} frames -> {os.path.relpath(path, frozen.REPO)} ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
