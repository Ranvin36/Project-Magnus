"""
Tune each baseline on validation, then evaluate the chosen settings once on test.

Selection (same for every method): highest validation F1 (4 px tolerance); ties ->
fewer false positives. Results are printed and saved to
data/<dataset>/derived/baseline_results_<tag>.json.

Usage: python -m tracking.run_baselines
"""
import itertools
import json
import os
import time

from tracking import baselines, frozen
from tracking.evaluate import fmt, load, reference_failures, score

DERIVED = os.path.join(frozen.REPO, "data", frozen.DATASET, "derived")

GRIDS = {
    "viterbi": dict(sigma=[6.0, 12.0, 24.0], c_none=[0.0, 1.0, 3.0], c_switch=[2.0, 4.0, 8.0],
                    w_vis=[0.5, 1.0, 2.0], interp_gap=[0]),
    "viterbi+interp": dict(sigma=[6.0, 12.0, 24.0], c_none=[0.0, 1.0, 3.0], c_switch=[2.0, 4.0, 8.0],
                           w_vis=[0.5, 1.0, 2.0], interp_gap=[5, 10, 20]),
    "kalman": dict(q=[1.0, 4.0, 16.0], r=[1.0, 4.0], gate=[9.21, 25.0], max_coast=[5, 10, 20],
                   init_vis=[0.6, 0.9], meas_vis=[0.1, 0.3]),
}
FUNCS = {"viterbi": baselines.viterbi, "viterbi+interp": baselines.viterbi, "kalman": baselines.kalman}


def run(fn, data, **kw):
    return {cid: fn(clip["frames"], **kw) for cid, clip in data["clips"].items()}


def main():
    val = load(os.path.join(DERIVED, f"candidates_{frozen.TAG}_val.json"))
    test = load(os.path.join(DERIVED, f"candidates_{frozen.TAG}_test.json"))
    fv, ft = reference_failures(val, frozen.VIS_THRESHOLD), reference_failures(test, frozen.VIS_THRESHOLD)
    results = {}

    fixed = {"per-frame (argmax)": (baselines.per_frame, dict(vis_threshold=frozen.VIS_THRESHOLD, decode="argmax")),
             "per-frame (centroid)": (baselines.per_frame, dict(vis_threshold=frozen.VIS_THRESHOLD, decode="centroid"))}
    chosen = dict(fixed)
    for name, grid in GRIDS.items():
        t0 = time.time()
        keys = list(grid)
        best = None
        for values in itertools.product(*(grid[k] for k in keys)):
            kw = dict(zip(keys, values))
            m = score(val, run(FUNCS[name], val, **kw), fv)
            key = (m["f1"], -m["false_pos"])
            if best is None or key > best[0]:
                best = (key, kw)
        chosen[name] = (FUNCS[name], best[1])
        print(f"tuned {name} on val over {len(list(itertools.product(*grid.values())))} settings in {time.time() - t0:.0f}s: {best[1]}", flush=True)

    for split, data, fails in (("val", val, fv), ("test", test, ft)):
        print(f"\n=== {split} ({sum(len(c['frames']) for c in data['clips'].values())} frames) ===")
        for name, (fn, kw) in chosen.items():
            m = score(data, run(fn, data, **kw), fails)
            results.setdefault(name, {"settings": kw})[split] = m
            print(fmt(name, m))

    path = os.path.join(DERIVED, f"baseline_results_{frozen.TAG}.json")
    with open(path, "w") as f:
        json.dump({"detector": frozen.TAG, "selection": "val F1 (4px), ties -> fewer false positives",
                   "results": {k: {**v, "settings": {kk: vv for kk, vv in v["settings"].items()}} for k, v in results.items()}}, f, indent=1)
    print(f"\nsaved -> {os.path.relpath(path, frozen.REPO)}")


if __name__ == "__main__":
    main()
