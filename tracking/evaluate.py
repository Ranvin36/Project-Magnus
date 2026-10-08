"""
Shared evaluation for every tracker (baselines and graph tracker), on the same
candidate files and the same definitions.

A method produces, per clip, one entry per frame: None ("no ball") or (x, y) in
512x288 pixels. Definitions (see docs/design/gnn_tracker_selection.md):
  hit         ball visible, method outputs a position within TOL px of the label
  F1          over hits / misses / false alarms, like the selection rule
  false pos.  share of "no visible ball" frames where the method outputs a position
  wrong obj.  share of visible frames where the output is > WRONG px from the label
  failure frames   visible frames where the reference per-frame detector (argmax,
                   threshold 0.6) misses or is > WRONG px off -- fixed once, so every
                   method is scored on recovering the same frames
"""
import json

import numpy as np

TOL = 4.0
WRONG = 10.0


def load(path):
    with open(path) as f:
        return json.load(f)


def _err(pos, gt):
    return float(np.hypot(pos[0] - gt[0], pos[1] - gt[1]))


def reference_failures(data, vis_threshold=0.6):
    """{clip: set(frame t)} where the per-frame argmax detector fails on a visible ball."""
    fails = {}
    for cid, clip in data["clips"].items():
        s = set()
        for f in clip["frames"]:
            if not f["visible"]:
                continue
            c = f["candidates"]
            if f["vis"] <= vis_threshold or not c or _err((c[0]["x"], c[0]["y"]), f["gt"]) > WRONG:
                s.add(f["t"])
        fails[cid] = s
    return fails


def score(data, outputs, failures=None):
    """outputs: {clip: [None | (x, y)] per frame}. Returns a dict of metrics."""
    failures = failures if failures is not None else reference_failures(data)
    tp = fn = fp = 0
    n_vis = n_novis = n_fp = n_wrong = n_out_vis = 0
    errs, fail_total, fail_hit, fail_errs = [], 0, 0, []
    for cid, clip in data["clips"].items():
        out = outputs[cid]
        for f, o in zip(clip["frames"], out):
            if f["visible"]:
                n_vis += 1
                if o is None:
                    fn += 1
                else:
                    n_out_vis += 1
                    e = _err(o, f["gt"]); errs.append(e)
                    if e <= TOL:
                        tp += 1
                    else:
                        fn += 1; fp += 1
                    n_wrong += e > WRONG
                if f["t"] in failures[cid]:
                    fail_total += 1
                    if o is not None:
                        e = _err(o, f["gt"]); fail_errs.append(e)
                        fail_hit += e <= TOL
            else:
                n_novis += 1
                if o is not None:
                    fp += 1; n_fp += 1
    errs = np.array(errs) if errs else np.array([np.nan])
    return {
        "f1": 2 * tp / max(2 * tp + fp + fn, 1),
        "detected": n_out_vis / max(n_vis, 1),
        "false_pos": n_fp / max(n_novis, 1),
        "wrong_obj": n_wrong / max(n_vis, 1),
        "median": float(np.median(errs)), "p90": float(np.percentile(errs, 90)),
        "failures_recovered": fail_hit / max(fail_total, 1),
        "failure_frames": fail_total,
        "failure_median_err": float(np.median(fail_errs)) if fail_errs else float("nan"),
    }


def fmt(name, m):
    return (f"{name:34} F1 {m['f1']:.3f} | det {100*m['detected']:5.1f}% | FP {100*m['false_pos']:5.1f}% | "
            f"wrong-obj {100*m['wrong_obj']:4.1f}% | med {m['median']:4.2f} p90 {m['p90']:5.2f} | "
            f"recovered {100*m['failures_recovered']:5.1f}% of {m['failure_frames']} failure frames")
