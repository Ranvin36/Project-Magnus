"""
Baseline trackers. Each takes one clip's frames (from a candidates file) and returns,
per frame, None ("no ball") or (x, y) in 512x288 pixels.

  per_frame   no time: the strongest candidate if the detector's visibility > threshold
  viterbi     most consistent chain through candidates + a "no ball" state per frame
              (dynamic programming over the whole clip), optionally filling short
              interior gaps by linear interpolation
  kalman      constant-acceleration Kalman filter that coasts through gaps, then an
              RTS smoother over each track; interior gaps (measured frames on both
              sides) are filled with smoothed positions

Viterbi and Kalman both use frames on both sides of a gap, as the graph tracker's
window does, so the comparison is fair.
"""
import numpy as np


def _pos(c, decode):
    return (c["cx"], c["cy"]) if decode == "centroid" else (float(c["x"]), float(c["y"]))


# ---------------------------------------------------------------- per-frame
def per_frame(frames, vis_threshold=0.6, decode="argmax"):
    return [(_pos(f["candidates"][0], decode) if f["vis"] > vis_threshold and f["candidates"] else None) for f in frames]


# ---------------------------------------------------------------- viterbi
def viterbi(frames, sigma=12.0, max_jump=60.0, min_score=0.1, w_vis=1.0, c_none=1.0,
            c_switch=4.0, interp_gap=0, decode="centroid"):
    """States per frame: candidates with score >= min_score, plus NONE (last index)."""
    eps = 1e-6
    states, emit = [], []
    for f in frames:
        cs = [c for c in f["candidates"] if c["score"] >= min_score]
        pos = np.array([_pos(c, decode) for c in cs], dtype=float).reshape(-1, 2)
        e = [-np.log(max(c["score"], eps)) - w_vis * np.log(max(f["vis"], eps)) for c in cs]
        e.append(-w_vis * np.log(max(1 - f["vis"], eps)) + c_none)
        states.append(pos); emit.append(np.array(e))

    def trans(p_prev, p_cur):
        n0, n1 = len(p_prev), len(p_cur)
        T = np.full((n0 + 1, n1 + 1), np.inf)
        if n0 and n1:
            d = np.linalg.norm(p_prev[:, None, :] - p_cur[None, :, :], axis=2)
            T[:n0, :n1] = np.where(d <= max_jump, 0.5 * (d / sigma) ** 2, np.inf)
        T[:n0, n1] = c_switch      # ball -> none
        T[n0, :n1] = c_switch      # none -> ball
        T[n0, n1] = 0.0            # none -> none
        return T

    cost, back = emit[0].copy(), []
    for t in range(1, len(frames)):
        tot = cost[:, None] + trans(states[t - 1], states[t])
        back.append(np.argmin(tot, axis=0))
        cost = tot[back[-1], np.arange(tot.shape[1])] + emit[t]
    path = [int(np.argmin(cost))]
    for b in reversed(back):
        path.append(int(b[path[-1]]))
    path.reverse()
    out = [tuple(states[t][s]) if s < len(states[t]) else None for t, s in enumerate(path)]
    if interp_gap:
        out = _interpolate(out, interp_gap)
    return out


def _interpolate(out, max_gap):
    out = list(out)
    known = [t for t, o in enumerate(out) if o is not None]
    for a, b in zip(known, known[1:]):
        if 1 < b - a <= max_gap + 1:
            for t in range(a + 1, b):
                w = (t - a) / (b - a)
                out[t] = (out[a][0] * (1 - w) + out[b][0] * w, out[a][1] * (1 - w) + out[b][1] * w)
    return out


# ---------------------------------------------------------------- kalman
_F = np.array([[1, 0, 1, 0, .5, 0], [0, 1, 0, 1, 0, .5], [0, 0, 1, 0, 1, 0],
               [0, 0, 0, 1, 0, 1], [0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 1]], dtype=float)
_H = np.array([[1, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0]], dtype=float)


def kalman(frames, q=4.0, r=2.0, gate=9.21, max_coast=10, init_vis=0.6, meas_vis=0.3,
           min_score=0.1, decode="centroid", width=512, height=288):
    """Constant-acceleration Kalman + RTS smoother; one track at a time, re-initialised when lost."""
    n = len(frames)
    Q = np.diag([q / 4, q / 4, q / 2, q / 2, q, q])
    R = np.eye(2) * r
    out = [None] * n
    t, track = 0, None   # track: list of (t, x_pred, P_pred, x_filt, P_filt, measured)
    segments = []

    def close(tr):
        if tr and any(m for *_, m in tr):
            segments.append(tr)

    for t in range(n):
        f = frames[t]
        cands = [c for c in f["candidates"] if c["score"] >= min_score]
        if track is None:
            if f["vis"] > init_vis and cands:
                z = np.array(_pos(cands[0], decode))
                x = np.array([z[0], z[1], 0, 0, 0, 0], dtype=float)
                P = np.diag([r, r, 400.0, 400.0, 100.0, 100.0])
                track = [(t, x.copy(), P.copy(), x, P, True)]
                coast = 0
            continue
        x_prev, P_prev = track[-1][3], track[-1][4]
        x_pred = _F @ x_prev
        P_pred = _F @ P_prev @ _F.T + Q
        S = _H @ P_pred @ _H.T + R
        Si = np.linalg.inv(S)
        best, best_cost = None, np.inf
        if f["vis"] > meas_vis:
            for c in cands:
                z = np.array(_pos(c, decode))
                y = z - _H @ x_pred
                d2 = float(y @ Si @ y)
                if d2 < gate and d2 - np.log(max(c["score"], 1e-6)) < best_cost:
                    best, best_cost = z, d2 - np.log(max(c["score"], 1e-6))
        if best is not None:
            K = P_pred @ _H.T @ Si
            x_f = x_pred + K @ (best - _H @ x_pred)
            P_f = (np.eye(6) - K @ _H) @ P_pred
            track.append((t, x_pred, P_pred, x_f, P_f, True)); coast = 0
        else:
            coast += 1
            outside = not (-20 <= x_pred[0] <= width + 20 and -20 <= x_pred[1] <= height + 20)
            if coast > max_coast or outside:
                close(track); track = None
                # this frame may start a new track
                if f["vis"] > init_vis and cands:
                    z = np.array(_pos(cands[0], decode))
                    x = np.array([z[0], z[1], 0, 0, 0, 0], dtype=float)
                    P = np.diag([r, r, 400.0, 400.0, 100.0, 100.0])
                    track = [(t, x.copy(), P.copy(), x, P, True)]; coast = 0
                continue
            track.append((t, x_pred, P_pred, x_pred, P_pred, False))
    close(track)

    for tr in segments:
        # RTS smoother
        xs = [s[3] for s in tr]; Ps = [s[4] for s in tr]
        for k in range(len(tr) - 2, -1, -1):
            x_pred_next, P_pred_next = tr[k + 1][1], tr[k + 1][2]
            C = Ps[k] @ _F.T @ np.linalg.inv(P_pred_next)
            xs[k] = xs[k] + C @ (xs[k + 1] - x_pred_next)
            Ps[k] = Ps[k] + C @ (Ps[k + 1] - P_pred_next) @ C.T
        measured = [s[5] for s in tr]
        last_meas = max(i for i, m in enumerate(measured) if m)
        for i, s in enumerate(tr):
            if i <= last_meas:          # measured frames and interior gaps only
                out[s[0]] = (float(xs[i][0]), float(xs[i][1]))
    return out
