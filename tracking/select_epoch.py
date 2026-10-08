"""
Pick a run's best epoch snapshot and visibility threshold on the validation split,
with the same pre-registered rule that chose the frozen detector:

  highest validation F1 (4 px tolerance) of the per-frame detector (argmax decoding),
  over all epoch snapshots x thresholds {0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.98, 0.99, 0.995, 0.999}

Writes checkpoints/<run>/<run>_selection.json and prints the per-epoch table.

Usage: python -m tracking.select_epoch --run tennis_foldA --splits-file splits_foldA.json
"""
import argparse
import glob
import json
import os
import re

import torch

from tracking import baselines, frozen
from tracking.evaluate import score
from tracking.extract_candidates import DATA, candidates_for_clips

THRESHOLDS = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.98, 0.99, 0.995, 0.999]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True)
    p.add_argument("--splits-file", default="splits.json")
    p.add_argument("--epochs", type=int, nargs="*", help="default: every snapshot found")
    args = p.parse_args()

    run_dir = os.path.join(frozen.REPO, "checkpoints", args.run)
    epochs = args.epochs or sorted(int(re.search(r"_epoch(\d+)\.pt$", f).group(1))
                                   for f in glob.glob(os.path.join(run_dir, f"{args.run}_epoch*.pt")))
    val_ids = json.load(open(os.path.join(DATA, args.splits_file)))["val"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    table, best = [], None
    for ep in epochs:
        model = frozen.load_detector(device, run=args.run, epoch=ep)
        data = {"clips": candidates_for_clips(model, val_ids, device)}
        row = {"epoch": ep}
        for thr in THRESHOLDS:
            m = score(data, {cid: baselines.per_frame(c["frames"], vis_threshold=thr, decode="argmax")
                             for cid, c in data["clips"].items()})
            row[str(thr)] = round(m["f1"], 4)
            if best is None or m["f1"] > best["val_f1"]:
                best = {"epoch": ep, "vis_threshold": thr, "val_f1": m["f1"],
                        "val_false_pos": m["false_pos"], "val_detected": m["detected"]}
        table.append(row)
        top = max(THRESHOLDS, key=lambda t: row[str(t)])
        print(f"epoch {ep:02d}: best val F1 {row[str(top)]:.3f} @ {top}", flush=True)

    out = {"run": args.run, "splits_file": args.splits_file,
           "rule": "max val F1 (4px), per-frame argmax decoding, thresholds " + ",".join(map(str, THRESHOLDS)),
           "selected": best, "table": table}
    path = os.path.join(run_dir, f"{args.run}_selection.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=1)
    print(f"\nSELECTED: epoch {best['epoch']} @ threshold {best['vis_threshold']} (val F1 {best['val_f1']:.3f}) -> {os.path.relpath(path, frozen.REPO)}")


if __name__ == "__main__":
    main()
