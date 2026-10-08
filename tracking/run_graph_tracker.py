"""
Train the graph tracker on simulated detector output, select on the real validation
match, and evaluate once on the real test matches -- next to the baselines.

  training data   real ball trajectories from the training matches, with real detector
                  behaviour transplanted from the validation match (tracking/simulate.py),
                  re-drawn every epoch
  selection       epoch and presence threshold with the highest F1 on the real validation
                  candidates (same scoring as the baselines; ties -> fewer false positives)
  test            once, on the real test candidates from the frozen detector

Usage: python -m tracking.run_graph_tracker [--epochs 40] [--seed 0]
"""
import argparse
import copy
import json
import os
import time

import numpy as np
import torch

from tracking import frozen, graph_tracker as gt, simulate
from tracking.evaluate import fmt, load, reference_failures, score

DERIVED = os.path.join(frozen.REPO, "data", frozen.DATASET, "derived")
LABELS = os.path.join(frozen.REPO, "data", frozen.DATASET, "512x288", "labels")
PRESENCE_THRESHOLDS = [0.3, 0.4, 0.5, 0.6, 0.7]


def evaluate(model, data, device, thr, fails):
    model.eval()
    out = {cid: gt.track_clip(model, c["frames"], device, thr) for cid, c in data["clips"].items()}
    return score(data, out, fails)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    args = p.parse_args()
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    val = load(os.path.join(DERIVED, f"candidates_{frozen.TAG}_val.json"))
    test = load(os.path.join(DERIVED, f"candidates_{frozen.TAG}_test.json"))
    fv, ft = reference_failures(val), reference_failures(test)
    train_ids = json.load(open(os.path.join(frozen.REPO, "data", frozen.DATASET, "512x288", "splits.json")))["train"]
    trajectories = simulate.load_trajectories(LABELS, train_ids)

    model = gt.GraphTracker().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    best = None
    for ep in range(1, args.epochs + 1):
        t0 = time.time()
        windows = [w for _, frames in trajectories
                   for _, w in gt.clip_windows(simulate.inject_long_gaps(simulate.transplant_clip(frames, val, rng), val, rng))]
        order = rng.permutation(len(windows))
        model.train(); total = 0.0
        for i in range(0, len(order), args.batch):
            b = {k: v.to(device) for k, v in gt.collate([windows[j] for j in order[i:i + args.batch]]).items()}
            loss = gt.loss_fn(model(b), b)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); total += loss.item()
        sched.step()
        m = evaluate(model, val, device, 0.5, fv)
        key = (m["f1"], -m["false_pos"])
        tag = ""
        if best is None or key > best[0]:
            best = (key, ep, copy.deepcopy(model.state_dict())); tag = "  <- best"
        print(f"epoch {ep:02d} loss {total / max(1, len(order) // args.batch):.3f} | val F1 {m['f1']:.3f} FP {100*m['false_pos']:.1f}% "
              f"recovered {100*m['failures_recovered']:.1f}% ({time.time() - t0:.0f}s){tag}", flush=True)

    model.load_state_dict(best[2])
    thr_best = max(PRESENCE_THRESHOLDS, key=lambda t: (lambda m: (m["f1"], -m["false_pos"]))(evaluate(model, val, device, t, fv)))
    print(f"\nselected: epoch {best[1]}, presence threshold {thr_best}")
    out_dir = os.path.join(frozen.REPO, "checkpoints", f"graph_tracker_v2_{frozen.TAG}_seed{args.seed}")
    os.makedirs(out_dir, exist_ok=True)
    torch.save(best[2], os.path.join(out_dir, "graph_tracker.pt"))

    base = json.load(open(os.path.join(DERIVED, f"baseline_results_{frozen.TAG}.json")))["results"]
    results = {"settings": {"window": gt.WINDOW, "epoch": best[1], "presence_threshold": thr_best, "seed": args.seed, "epochs": args.epochs}}
    for split, data, fails in (("val", val, fv), ("test", test, ft)):
        print(f"\n=== {split} ===")
        for name in ("per-frame (argmax)", "viterbi", "viterbi+interp", "kalman"):
            print(fmt(name, base[name][split]))
        m = evaluate(model, data, device, thr_best, fails)
        results[split] = m
        print(fmt("GRAPH TRACKER", m))
    with open(os.path.join(out_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=1)
    print(f"\nsaved -> {os.path.relpath(out_dir, frozen.REPO)}")


if __name__ == "__main__":
    main()
