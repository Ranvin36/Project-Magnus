"""
Simulated detector output for training the graph tracker, so it needs no extra detectors.

1. fit_noise_model(): measures how the frozen detector behaves on real footage it never
   trained on -- the validation match (never the test matches):
     - for visible balls: a 2-state Markov chain "detected" <-> "blind" (blind = no
       candidate within TOL px of the ball), which reproduces failure *runs*
     - offsets and scores of the true candidate when detected
     - how many distractor candidates appear per frame, their scores, and how often a
       distractor persists across frames (static objects like logos, line judges)
     - visibility-probability distributions per situation
2. simulate_clip(): replays those statistics on a real ball trajectory (labels from the
   training matches) and returns frames in the same format as a candidates file.

The test matches are never used here.
"""
import json
import os

import numpy as np

from tracking.evaluate import TOL, WRONG

W, H = 512, 288


def _hit(c, gt):
    return gt is not None and np.hypot(c["x"] - gt[0], c["y"] - gt[1]) <= TOL


def fit_noise_model(cand_data):
    det_det = det_blind = blind_det = blind_blind = 0
    offsets, true_scores, true_areas = [], [], []
    n_distr, distr_scores, distr_areas = [], [], []
    persist_hits = persist_total = 0
    vis = {"detected": [], "blind": [], "absent": []}
    for clip in cand_data["clips"].values():
        prev_state, prev_distr = None, []
        for f in clip["frames"]:
            cs = f["candidates"]
            true = [c for c in cs if f["visible"] and _hit(c, f["gt"])]
            distr = [c for c in cs if c not in true]
            n_distr.append(len(distr))
            for c in distr:
                distr_scores.append(c["score"]); distr_areas.append(c["area"])
                persist_total += 1
                persist_hits += any(np.hypot(c["x"] - p["x"], c["y"] - p["y"]) <= 6 for p in prev_distr)
            prev_distr = distr
            if f["visible"]:
                state = "detected" if true else "blind"
                if true:
                    t = true[0]
                    offsets.append((t["x"] - f["gt"][0], t["y"] - f["gt"][1]))
                    true_scores.append(t["score"]); true_areas.append(t["area"])
                vis[state].append(f["vis"])
                if prev_state == "detected":
                    det_det += state == "detected"; det_blind += state == "blind"
                elif prev_state == "blind":
                    blind_det += state == "detected"; blind_blind += state == "blind"
                prev_state = state
            else:
                vis["absent"].append(f["vis"])
                prev_state = None
    return {
        "p_det_to_blind": det_blind / max(det_det + det_blind, 1),
        "p_blind_to_blind": blind_blind / max(blind_det + blind_blind, 1),
        "offsets": offsets, "true_scores": true_scores, "true_areas": true_areas,
        "n_distractors": n_distr, "distr_scores": distr_scores, "distr_areas": distr_areas,
        "p_distr_persist": persist_hits / max(persist_total, 1),
        "vis": vis,
    }


def summary(m):
    blind_frac = m["p_det_to_blind"] / (m["p_det_to_blind"] + 1 - m["p_blind_to_blind"])
    return (f"P(detected->blind) {m['p_det_to_blind']:.3f} | P(blind->blind) {m['p_blind_to_blind']:.3f} "
            f"(mean blind run {1 / max(1 - m['p_blind_to_blind'], 1e-6):.1f} frames, stationary blind share {100 * blind_frac:.1f}%)\n"
            f"true-candidate offset: median |d| {np.median(np.hypot(*np.array(m['offsets']).T)):.2f}px | "
            f"distractors/frame: mean {np.mean(m['n_distractors']):.2f}, zero in {100 * np.mean(np.array(m['n_distractors']) == 0):.0f}% | "
            f"distractor persists to next frame: {100 * m['p_distr_persist']:.0f}%\n"
            f"vis prob median -- detected {np.median(m['vis']['detected']):.3f}, blind {np.median(m['vis']['blind']):.3f}, "
            f"no ball {np.median(m['vis']['absent']):.3f}")


def transplant_clip(frames, source, rng):
    """
    Replay a real stretch of detector output (source = a candidates file the detector never
    trained on, e.g. the validation match) onto a real training trajectory.

    The training clip is walked alongside a random contiguous stretch of a random source
    clip, so failure runs, distractors, hallucinated blobs and their correlations are real:
      - training ball visible, source ball visible: the source's true candidate (if any) is
        moved to the training ball, keeping its real offset/score/area; everything else in
        the source frame is copied as a distractor; visibility prob copied
      - training ball not visible: a real "no ball" source frame is copied as-is
      - when the source frame's situation differs, the next source frame of the right
        kind is used, so the source stretch still advances in order
    """
    clips = [c["frames"] for c in source["clips"].values()]
    absent_pool = [f for c in clips for f in c if not f["visible"]]
    seq = clips[int(rng.integers(len(clips)))]
    i = int(rng.integers(len(seq)))
    out = []
    for f in frames:
        want_visible = f["visible"] and f["gt"] is not None
        if want_visible:
            for _ in range(len(seq)):
                vf = seq[i % len(seq)]; i += 1
                if vf["visible"] and vf["gt"] is not None:
                    break
            cands = []
            for c in vf["candidates"]:
                # candidates within WRONG px belong to the ball (incl. streak offsets): move them with it
                if np.hypot(c["x"] - vf["gt"][0], c["y"] - vf["gt"][1]) <= WRONG:
                    dx, dy = c["x"] - vf["gt"][0], c["y"] - vf["gt"][1]
                    dcx, dcy = c["cx"] - vf["gt"][0], c["cy"] - vf["gt"][1]
                    x, y = np.clip(f["gt"][0] + dx, 0, W - 1), np.clip(f["gt"][1] + dy, 0, H - 1)
                    cands.append({**c, "x": int(round(x)), "y": int(round(y)),
                                  "cx": float(np.clip(f["gt"][0] + dcx, 0, W - 1)), "cy": float(np.clip(f["gt"][1] + dcy, 0, H - 1))})
                else:
                    cands.append(dict(c))
        else:
            vf = seq[i % len(seq)]; i += 1
            if vf["visible"]:
                vf = absent_pool[int(rng.integers(len(absent_pool)))]
            cands = [dict(c) for c in vf["candidates"]]
        cands.sort(key=lambda c: -c["score"])
        out.append({**f, "vis": vf["vis"], "candidates": cands})
    return out


def load_trajectories(label_dir, clip_ids):
    """Real ball trajectories (512x288) from label files: list of (clip_id, frames)."""
    out = []
    for cid in clip_ids:
        meta = json.load(open(os.path.join(label_dir, cid + ".json")))
        iw, ih = meta["camera"]["intrinsics"]["width"], meta["camera"]["intrinsics"]["height"]
        frames = []
        for t, e in enumerate(meta["frames"]):
            gt = [e["centre_px"][0] * W / iw, e["centre_px"][1] * H / ih] if e["centre_px"] else None
            frames.append({"t": t, "gt": gt, "visible": bool(e["visible"]), "occluded": bool(e["occluded"]),
                           "hard": bool(e.get("hard"))})
        out.append((cid, frames))
    return out


def simulate_clip(frames, m, rng):
    """Replay the noise model on one real trajectory -> frames in candidates-file format."""
    out, state, static = [], "detected", []
    n_off = len(m["offsets"])
    for f in frames:
        cands = []
        # distractors: keep some from the previous frame (static objects), add new ones
        n = int(rng.choice(m["n_distractors"]))
        kept = [d for d in static if rng.random() < m["p_distr_persist"]][:n]
        new = [{"x": int(rng.integers(0, W)), "y": int(rng.integers(0, H))} for _ in range(n - len(kept))]
        static = kept + new
        for d in static:
            j = int(rng.integers(len(m["distr_scores"])))
            cands.append({"x": d["x"], "y": d["y"], "cx": float(d["x"]), "cy": float(d["y"]),
                          "score": float(m["distr_scores"][j]), "area": int(m["distr_areas"][j])})
        if f["visible"] and f["gt"] is not None:
            p_blind = m["p_blind_to_blind"] if state == "blind" else m["p_det_to_blind"]
            state = "blind" if rng.random() < p_blind else "detected"
            if state == "detected":
                j = int(rng.integers(n_off))
                dx, dy = m["offsets"][j]
                x, y = float(np.clip(f["gt"][0] + dx, 0, W - 1)), float(np.clip(f["gt"][1] + dy, 0, H - 1))
                cands.append({"x": int(round(x)), "y": int(round(y)), "cx": x, "cy": y,
                              "score": float(m["true_scores"][j]), "area": int(m["true_areas"][j])})
            vis = float(rng.choice(m["vis"][state]))
        else:
            state = "detected"
            vis = float(rng.choice(m["vis"]["absent"]))
        cands.sort(key=lambda c: -c["score"])
        out.append({**f, "vis": vis, "candidates": cands[:8]})
    return out


def inject_long_gaps(frames, source, rng, per_100_frames=0.6, min_len=6, max_len=25):
    """Remove the true ball's candidates for long stretches (detector-blind runs longer than
    the validation match contains), with visibility probs drawn from real blind frames."""
    blind_vis = [f["vis"] for c in source["clips"].values() for f in c["frames"]
                 if f["visible"] and f["gt"] is not None and not any(_hit(x, f["gt"]) for x in f["candidates"])]
    out = [dict(f) for f in frames]
    n_gaps = rng.poisson(per_100_frames * len(frames) / 100)
    for _ in range(n_gaps):
        L = int(rng.integers(min_len, max_len + 1))
        s0 = int(rng.integers(0, max(1, len(out) - L)))
        for t in range(s0, min(s0 + L, len(out))):
            f = out[t]
            if f["visible"] and f["gt"] is not None:
                f["candidates"] = [c for c in f["candidates"]
                                   if np.hypot(c["x"] - f["gt"][0], c["y"] - f["gt"][1]) > WRONG]
                f["vis"] = float(rng.choice(blind_vis))
    return out
