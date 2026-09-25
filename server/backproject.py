"""
Step 1: turn each 2D detection into a rough 3D point, using the ball's size.

The further away the ball is, the smaller it looks:
    width_px = focal * BALL_DIAMETER / Z
so
    Z = focal * BALL_DIAMETER / width_px

3D points are in the camera's own frame (OpenCV convention), in metres:
    X = right, Y = down, Z = forward (distance in front of the camera)

Usage (checks it against a cricket-synth label that has the true depth):
    python server/backproject.py path/to/label.json
"""
import json
import sys

import numpy as np

BALL_DIAMETER = 0.0716  # metres, cricket ball (same value cricket-synth renders)


def backproject(u, v, width_px, focal, cx, cy):
    """Pixel position + pixel width of the ball -> 3D point (X, Y, Z) in metres.

    u, v      ball centre in pixels
    width_px  ball width (diameter) in pixels
    focal     focal length in pixels
    cx, cy    image centre in pixels
    Works on single numbers or numpy arrays (one entry per frame).
    """
    Z = focal * BALL_DIAMETER / width_px
    X = (u - cx) * Z / focal
    Y = (v - cy) * Z / focal
    return np.stack([X, Y, Z], axis=-1)


def check_on_label(label_path):
    with open(label_path) as f:
        label = json.load(f)
    cam = label["camera"]["intrinsics"]
    frames = [fr for fr in label["frames"] if fr["visible"]]

    u = np.array([fr["centre_px"][0] for fr in frames])
    v = np.array([fr["centre_px"][1] for fr in frames])
    width = np.array([2 * fr["radius_px"] for fr in frames])
    true_depth = np.array([fr["depth_m"] for fr in frames])

    points = backproject(u, v, width, cam["fx"], cam["cx"], cam["cy"])

    print(f"{'frame':>5} {'width px':>9} {'Z ours':>8} {'Z true':>8} {'error':>7}")
    for fr, w, p, z in zip(frames, width, points, true_depth):
        print(f"{fr['frame']:5d} {w:9.1f} {p[2]:8.2f} {z:8.2f} {p[2] - z:+7.2f}")
    err = np.abs(points[:, 2] - true_depth)
    print(f"\nmean depth error {err.mean():.3f} m, worst {err.max():.3f} m")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    check_on_label(sys.argv[1])
