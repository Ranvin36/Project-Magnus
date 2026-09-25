"""
Measure how wide the ball looks (in pixels) at each detection.

TrackNet only says where the ball is, not how big it looks, and the 3D stage
needs the size to work out distance.

How: take a small patch around the detected ball, and the same patch a few
frames earlier and later, when the ball was somewhere else. The pixels that
differ from both are the ball. A fast ball is blurred into a streak, which is
longer along its motion, but its short side is still the ball's diameter, so
we take the short side of the changed blob.
"""
import cv2
import numpy as np

HALF = 40       # patch is 80x80 px around the ball
GAP = 3         # compare with the frames 3 before and 3 after
THRESHOLD = 25  # grey-level change that counts as "different"


def _patch(gray, u, v):
    h, w = gray.shape
    x0, y0 = int(round(u)) - HALF, int(round(v)) - HALF
    if x0 < 0 or y0 < 0 or x0 + 2 * HALF > w or y0 + 2 * HALF > h:
        return None                                   # too close to the edge
    return gray[y0:y0 + 2 * HALF, x0:x0 + 2 * HALF].astype(np.int16)


def _width_from_patches(ball, others):
    # Keep only what differs from every other frame: the ball, not its old/new spot.
    diff = np.min([np.abs(ball - o) for o in others], axis=0).astype(np.uint8)
    mask = (cv2.GaussianBlur(diff, (3, 3), 0) > THRESHOLD).astype(np.uint8)
    count, labels, stats, centres = cv2.connectedComponentsWithStats(mask)
    if count < 2:
        return None
    # The blob closest to the patch centre (where TrackNet put the ball).
    dist = np.linalg.norm(centres[1:] - HALF, axis=1)
    blob = 1 + int(np.argmin(dist))
    if dist[blob - 1] > 10 or stats[blob, cv2.CC_STAT_AREA] < 4:
        return None
    ys, xs = np.nonzero(labels == blob)
    (_, _), (a, b), _ = cv2.minAreaRect(np.stack([xs, ys], axis=1).astype(np.float32))
    return float(min(a, b)) + 1.0     # +1: pixel centres to pixel edges


def measure_widths(video_path, detections):
    """detections: list per frame of (u, v, conf) or None.
    Returns a list per frame of the ball width in pixels, or None if it
    couldn't be measured (no ball, near the edge, or nothing clear changed)."""
    n = len(detections)
    # Which frames we need to cut patches from, for which detection.
    wanted = {}
    for i, d in enumerate(detections):
        if d is not None:
            for j in (i - GAP, i, i + GAP):
                if 0 <= j < n:
                    wanted.setdefault(j, []).append(i)

    patches = {}                                       # (detection i, frame j) -> patch
    cap = cv2.VideoCapture(video_path)
    for j in range(n):
        ok, frame = cap.read()
        if not ok:
            break
        if j in wanted:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            for i in wanted[j]:
                patches[(i, j)] = _patch(gray, *detections[i][:2])
    cap.release()

    widths = [None] * n
    for i, d in enumerate(detections):
        ball = patches.get((i, i))
        others = [patches.get((i, j)) for j in (i - GAP, i + GAP)]
        others = [o for o in others if o is not None]
        if ball is not None and others:
            widths[i] = _width_from_patches(ball, others)
    return widths
