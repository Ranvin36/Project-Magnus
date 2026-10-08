"""
Turn one detector heatmap into a short list of ball candidates.

One candidate per blob (connected region above BLOB_THRESHOLD). Per-pixel local
maxima don't work here: the detector's heatmaps saturate to flat-topped blobs, so
one blob would yield many "peaks" and crowd every other blob out of the top-K.

Each candidate records, in 512x288 pixels:
  x, y    strongest pixel of the blob
  cx, cy  intensity-weighted centre of the blob's core (pixels >= half its peak) --
          a sub-pixel estimate, and closer to the streak middle for blurred balls
  score   the blob's peak heatmap value
  area    core size in pixels (large = blurred / elongated / uncertain)
"""
import cv2
import numpy as np

BLOB_THRESHOLD = 0.05
MAX_CANDIDATES = 8


def extract_candidates(heatmap, blob_threshold=BLOB_THRESHOLD, max_candidates=MAX_CANDIDATES):
    """heatmap: (H, W) float array of probabilities -> list of candidate dicts, strongest first."""
    n_labels, labels = cv2.connectedComponents((heatmap > blob_threshold).astype(np.uint8), connectivity=8)
    candidates = []
    for k in range(1, n_labels):
        ys, xs = np.nonzero(labels == k)
        vals = heatmap[ys, xs]
        i = int(np.argmax(vals))
        peak = float(vals[i])
        core = vals >= 0.5 * peak
        w = vals[core]
        candidates.append({
            "x": int(xs[i]), "y": int(ys[i]),
            "cx": float((xs[core] * w).sum() / w.sum()), "cy": float((ys[core] * w).sum() / w.sum()),
            "score": peak, "area": int(core.sum()),
        })
    candidates.sort(key=lambda c: -c["score"])
    return candidates[:max_candidates]
