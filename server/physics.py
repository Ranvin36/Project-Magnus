"""
The physics: how a cricket ball moves through the air.

    acceleration = gravity + drag

Drag pushes against the motion and grows with speed squared. Spin (Magnus) is
left out for now: from one camera it is too weak to measure reliably, and for a
cricket ball it moves the path by only a few centimetres.

Gravity is passed in as a vector because we work in the camera's frame, where
"down" is whatever direction the fit finds (the camera may be tilted).
"""
import numpy as np

from backproject import BALL_DIAMETER

MASS = 0.160                # kg
AIR_DENSITY = 1.225         # kg/m^3
DRAG_COEFF = 0.45
GRAVITY = 9.81              # m/s^2

AREA = np.pi * (BALL_DIAMETER / 2) ** 2
DRAG = AIR_DENSITY * DRAG_COEFF * AREA / (2 * MASS)   # ~0.0069 per metre


def _run(position, velocity, gravity, t0, targets, dt):
    """Step from t0 through the sorted targets. dt < 0 runs the clock backwards.

    Each small step:  velocity += acceleration * h,  position += velocity * h
    with  acceleration = gravity - DRAG * speed * velocity.
    Written with plain numbers instead of numpy: for 3-number vectors that is
    several times faster, and the fit calls this thousands of times.
    """
    x, y, z = map(float, position)
    vx, vy, vz = map(float, velocity)
    gx, gy, gz = map(float, gravity)
    t = t0
    out = []
    for target in targets:
        while abs(target - t) > 1e-9:
            h = dt if abs(target - t) > abs(dt) else target - t
            drag = DRAG * (vx * vx + vy * vy + vz * vz) ** 0.5
            vx += (gx - drag * vx) * h
            vy += (gy - drag * vy) * h
            vz += (gz - drag * vz) * h
            x += vx * h
            y += vy * h
            z += vz * h
            t += h
        out.append(((x, y, z), (vx, vy, vz)))
    return out


def simulate(position, velocity, gravity, t0, times, dt=0.002):
    """The ball is at `position` moving at `velocity` at time t0.
    Returns (positions, velocities), each (N, 3), at the given times.
    Times before t0 are fine: the flight is run backwards to reach them."""
    position = np.asarray(position, float)
    velocity = np.asarray(velocity, float)
    times = np.asarray(times, float)
    after = np.sort(times[times >= t0])
    before = np.sort(times[times < t0])[::-1]

    states = {}
    for tt, s in zip(after, _run(position, velocity, gravity, t0, after, dt)):
        states[tt] = s
    for tt, s in zip(before, _run(position, velocity, gravity, t0, before, -dt)):
        states[tt] = s
    positions = np.array([states[tt][0] for tt in times]).reshape(-1, 3)
    velocities = np.array([states[tt][1] for tt in times]).reshape(-1, 3)
    return positions, velocities
