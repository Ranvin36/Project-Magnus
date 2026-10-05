"""
Print one row per training run from checkpoints/*_summary.json (written at
the end of every notebook training run), best val_loss first.

Usage: python training/compare_runs.py
"""
import glob
import json
import os

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

rows = []
for path in glob.glob(os.path.join(REPO, "checkpoints", "*_summary.json")):
    with open(path) as f:
        s = json.load(f)
    h, c = s["history"], s["config"]
    if not h.get("val_loss"):
        continue
    best = min(range(len(h["val_loss"])), key=h["val_loss"].__getitem__)
    rows.append((s["best_val_loss"], c["RUN_NAME"], c["BATCH_SIZE"], c["LR"], c["VIS_LOSS_WEIGHT"],
                 s["global_step"], 100 * h["val_detect_rate"][best], 100 * h["val_fp_rate"][best],
                 h["val_err_px"][best], s["stop_reason"]))

print(f"{'run':28} {'bs':>3} {'lr':>8} {'vis_w':>5} {'steps':>6} {'val_loss':>8} "
      f"{'detect%':>7} {'FP%':>5} {'err_px':>6}  stop reason")
for loss, name, bs, lr, vw, steps, det, fp, err, reason in sorted(rows):
    print(f"{name:28} {bs:>3} {lr:>8.1e} {vw:>5} {steps:>6} {loss:>8.4f} {det:>7.1f} {fp:>5.1f} {err:>6.2f}  {reason}")
