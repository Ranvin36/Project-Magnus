"""
2D detections -> 3D flight, using only the ball's size and physics.

Step 1  back-project: ball size gives distance -> a rough 3D point per frame
Step 2  split:        cut the track where the ball bounces (or is hit)
Step 3  fit:          find the physically possible flight that best matches
                      the measured pixel positions and ball widths
Step 4  report:       3D positions for every frame, speed, fit quality

Everything is in the camera's frame (X right, Y down, Z forward, metres) until
step 4, which also rotates the result so that "up" is up, using the gravity
direction the fit found.
"""
import numpy as np
from scipy.optimize import least_squares

import physics
from backproject import BALL_DIAMETER, backproject

PIXEL_NOISE = 20.0  # how far off (px) we expect the detected ball centre to be
                    # (measured: the current tracker is off by ~20-30 px)
WIDTH_NOISE = 1.0   # how far off (px) we expect the measured ball width to be

# Most cameras are roughly level, so gravity points roughly "down the image".
# The fit may tilt it, but is gently pulled back (typical tilt ~15 degrees)
# and never beyond 45 degrees. Without this, noisy detections can turn "down"
# upside down, and every height comes out wrong.
TYPICAL_TILT = np.radians(15)
MAX_TILT = np.radians(45)

# A cut at a suspected bounce is kept only if it improves the fit by at least
# half AND by this much (in units of the expected noise).
MIN_GAIN = 2


# --- Step 2: split into flights --------------------------------------------
def split_flights(t, uv, width, focal, cx, cy, min_points=5):
    """Return a list of index arrays, one per free flight.

    Try cutting at every gap between two detections. A cut is kept when two
    separate physics flights fit the detections far better than one flight
    does: that's a bounce (or a hit). Repeats until no cut helps, so any
    number of bounces is handled.
    """
    costs = {}

    def cost(seg):
        key = (seg[0], seg[-1])
        if key not in costs:
            costs[key] = fit(t, uv, width, focal, cx, cy, [seg])[1]
        return costs[key]

    flights = [np.arange(len(t))]
    while True:
        best = None                             # (improvement, which flight, cut)
        for k, seg in enumerate(flights):
            if len(seg) < 2 * min_points:
                continue
            whole = cost(seg)
            for i in range(len(seg) - 1):
                left, right = seg[:i + 1], seg[i + 1:]
                if min(len(left), len(right)) < min_points:
                    continue
                gain = whole - cost(left) - cost(right)
                # "far better" = removes at least half the error, and by a clear margin
                if gain > 0.5 * whole and gain > MIN_GAIN and (best is None or gain > best[0]):
                    best = (gain, k, i)
        if best is None:
            return flights
        _, k, i = best
        seg = flights[k]
        flights[k:k + 1] = [seg[:i + 1], seg[i + 1:]]


# --- Step 3: fit -------------------------------------------------------------
def gravity_vector(tilt, roll):
    """Gravity in the camera frame. tilt = roll = 0 means straight down the image."""
    return physics.GRAVITY * np.array([
        -np.cos(tilt) * np.sin(roll),
        np.cos(tilt) * np.cos(roll),
        np.sin(tilt),
    ])


def project(points, focal, cx, cy):
    """3D camera-frame points -> pixel position and ball width in pixels."""
    Z = np.maximum(points[:, 2], 0.1)
    u = focal * points[:, 0] / Z + cx
    v = focal * points[:, 1] / Z + cy
    width = focal * BALL_DIAMETER / Z
    return u, v, width


def starting_guess(t, uv, width, focal, cx, cy, flights):
    """Rough start position + velocity for each flight, from the step 1 points."""
    has_width = ~np.isnan(width)
    width = np.interp(t, t[has_width], width[has_width])   # fill gaps from neighbours
    rough = backproject(uv[:, 0], uv[:, 1], width, focal, cx, cy)

    guess = [0.0, 0.0]                          # gravity: tilt, roll
    for idx in flights:
        dt = t[idx] - t[idx[0]]
        # straight-line fit through the rough points: position = start + velocity * dt
        velocity, start = np.polyfit(dt, rough[idx], 1)
        guess += list(start) + list(velocity)
    return np.array(guess)


def fit(t, uv, width, focal, cx, cy, flights):
    """Find gravity direction + each flight's start position/velocity so that
    the simulated flights, seen through the camera, match the detections.
    Returns the fitted numbers and how badly they still fit."""
    has_width = ~np.isnan(width)

    def residuals(x):
        gravity = gravity_vector(x[0], x[1])
        errors = []
        for k, idx in enumerate(flights):
            start, velocity = x[2 + 6 * k: 5 + 6 * k], x[5 + 6 * k: 8 + 6 * k]
            points, _ = physics.simulate(start, velocity, gravity, t[idx[0]], t[idx])
            u, v, w = project(points, focal, cx, cy)
            errors.append((u - uv[idx, 0]) / PIXEL_NOISE)
            errors.append((v - uv[idx, 1]) / PIXEL_NOISE)
            known = has_width[idx]
            errors.append((w[known] - width[idx][known]) / WIDTH_NOISE)
        errors.append(x[:2] / TYPICAL_TILT)     # the gentle pull towards image-down
        return np.concatenate(errors)

    x0 = starting_guess(t, uv, width, focal, cx, cy, flights)
    limits = np.full(len(x0), np.inf)
    limits[:2] = MAX_TILT                       # tilt and roll: at most 45 degrees
    # soft_l1 = a few bad detections can't drag the whole fit off
    result = least_squares(residuals, x0, loss="soft_l1", x_scale="jac",
                           bounds=(-limits, limits))
    return result.x, result.cost                # cost = how badly it still fits


# --- Step 4: report ----------------------------------------------------------
def reconstruct(t, uv, width, focal, cx, cy, fps):
    """t (N,) seconds, uv (N, 2) pixels, width (N,) pixels or NaN where unknown.
    Returns a dict ready to send to the frontend."""
    t, uv, width = np.asarray(t, float), np.asarray(uv, float), np.asarray(width, float)

    # A moving ball never sits on exactly the same pixel two frames running.
    # When it does, the tracker repeated itself (it does this for 3 frames at a
    # time), so keep only the first of each repeat: the others are fake data.
    repeat = np.zeros(len(t), bool)
    repeat[1:] = np.all(np.abs(np.diff(uv, axis=0)) < 0.5, axis=1)
    t, uv, width = t[~repeat], uv[~repeat], width[~repeat]

    has_width = ~np.isnan(width)
    if has_width.sum() < 2:
        raise ValueError("The ball width was measured in fewer than 2 frames, "
                         "so its distance can't be worked out.")
    # Frames without a width borrow one from their neighbours in time.
    filled = np.interp(t, t[has_width], width[has_width])

    flights = split_flights(t, uv, width, focal, cx, cy)            # step 2
    x, _ = fit(t, uv, width, focal, cx, cy, flights)                # step 3
    gravity = gravity_vector(x[0], x[1])
    starts = [(t[idx[0]], x[2 + 6 * k: 5 + 6 * k], x[5 + 6 * k: 8 + 6 * k])
              for k, idx in enumerate(flights)]

    def flight_at(k, times):
        t0, start, velocity = starts[k]
        return physics.simulate(start, velocity, gravity, t0, times)

    # Bounce = where one flight ends and the next begins: the moment between
    # their detections when the two fitted paths come closest.
    bounces = []
    for k in range(len(flights) - 1):
        times = np.linspace(t[flights[k][-1]], t[flights[k + 1][0]], 60)
        a, _ = flight_at(k, times)
        b, _ = flight_at(k + 1, times)
        i = np.argmin(np.linalg.norm(a - b, axis=1))
        bounces.append(dict(t=times[i], point=(a[i] + b[i]) / 2))

    # 3D position for every frame from the first to the last detection, gaps included.
    frames = np.arange(round(t[0] * fps), round(t[-1] * fps) + 1)
    times = frames / fps
    edges = [-np.inf] + [b["t"] for b in bounces] + [np.inf]
    points = np.zeros((len(frames), 3))
    for k in range(len(flights)):
        inside = (times >= edges[k]) & (times < edges[k + 1])
        if inside.any():
            points[inside], _ = flight_at(k, times[inside])
    u, v, _ = project(points, focal, cx, cy)

    # Rotate into a frame where up is up (from gravity) and forward is the
    # direction of travel; origin = where the ball was first seen.
    up = -gravity / np.linalg.norm(gravity)
    first_velocity = starts[0][2]
    forward = first_velocity - first_velocity.dot(up) * up
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, up)
    origin = points[0]
    if bounces:
        # The ball touches the ground at the bounce, so that sets ground level.
        ground = bounces[0]["point"].dot(up) - BALL_DIAMETER / 2
    else:
        ground = points.dot(up).min()           # unknown: use the lowest point
    def to_world(p):
        return np.stack([(p - origin).dot(forward), (p - origin).dot(right),
                         p.dot(up) - ground], axis=1)
    world = to_world(points)
    # The step 1 points (distance from ball size alone), in the same frame, to
    # show what the physics fit started from.
    rough = to_world(backproject(uv[:, 0], uv[:, 1], filled, focal, cx, cy))

    # How well the fitted path matches the detections (the confidence).
    fitted_u, fitted_v = np.interp(t, times, u), np.interp(t, times, v)
    pixel_error = np.hypot(fitted_u - uv[:, 0], fitted_v - uv[:, 1])

    metrics = dict(
        speed_first_seen_kmh=round(float(np.linalg.norm(first_velocity) * 3.6), 1),
        fit_error_px=round(float(np.sqrt(np.mean(pixel_error ** 2))), 2),
    )

    return dict(
        focal_px=focal,
        n_flights=len(flights),
        ground_known=bool(bounces),
        metrics=metrics,
        bounces=[dict(t=round(float(b["t"]), 4),
                      forward=round(float((b["point"] - origin).dot(forward)), 3),
                      right=round(float((b["point"] - origin).dot(right)), 3))
                 for b in bounces],
        frames=[dict(frame=int(f), u=round(float(uu), 1), v=round(float(vv), 1),
                     forward=round(float(p[0]), 3), right=round(float(p[1]), 3),
                     height=round(float(p[2]), 3))
                for f, uu, vv, p in zip(frames, u, v, world)],
        measured=[dict(frame=int(round(tt * fps)), forward=round(float(p[0]), 3),
                       right=round(float(p[1]), 3), height=round(float(p[2]), 3))
                  for tt, p in zip(t, rough)],
    )
