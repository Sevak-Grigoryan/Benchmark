"""Camera geometry shared by publishing/publish_episode.py and evaluate.py (pure Python, no numpy needed).

World ENU metres (x east, y north, z up). Camera yaw: 0 = +x, counter-clockwise positive. Camera pitch: positive = looking
down. Camera axes follow OpenCV (x right, y down, z forward):  X_cam = R_wc @ X_world + t_wc."""
import math


def rotation_wc(yaw_deg, pitch_deg):
    y, p = math.radians(yaw_deg), math.radians(pitch_deg)
    f = (math.cos(p) * math.cos(y), math.cos(p) * math.sin(y), -math.sin(p))   # forward
    r = (math.sin(y), -math.cos(y), 0.0)                                        # right
    d = (f[1] * r[2] - f[2] * r[1], f[2] * r[0] - f[0] * r[2], f[0] * r[1] - f[1] * r[0])  # down = forward x right
    return [list(r), list(d), list(f)]


def extrinsics(cam):
    R = rotation_wc(cam["yaw_deg"], cam["pitch_deg"])
    c = (cam["x"], cam["y"], cam["z"])
    t = [-sum(R[i][j] * c[j] for j in range(3)) for i in range(3)]
    return {"R_wc": [[round(v, 6) for v in row] for row in R], "t_wc": [round(v, 4) for v in t]}


def project(cam, point, K):
    """Pixel (u, v) of a world point seen from camera pose `cam`, or None if it is behind the camera."""
    R = rotation_wc(cam["yaw_deg"], cam["pitch_deg"])
    d = (point[0] - cam["x"], point[1] - cam["y"], point[2] - cam["z"])
    xc, yc, zc = (sum(R[i][j] * d[j] for j in range(3)) for i in range(3))
    if zc <= 0.1:
        return None
    return K["fx_px"] * xc / zc + K["cx_px"], K["fy_px"] * yc / zc + K["cy_px"]


def in_frame(uv, size, margin):
    """True if pixel uv lies inside the image with `margin` (fraction of width / height) kept clear at every edge."""
    if uv is None:
        return False
    w, h = size
    return margin * w <= uv[0] <= (1 - margin) * w and margin * h <= uv[1] <= (1 - margin) * h


def bearing_error_deg(pose, point):
    """Angle between the drone's heading and the horizontal direction to `point`."""
    want = math.atan2(point[1] - pose["y"], point[0] - pose["x"])
    e = want - math.radians(pose["yaw_deg"])
    return abs(math.degrees(math.atan2(math.sin(e), math.cos(e))))


def xyz(rec):
    p = rec["pose"]
    return (p["x"], p["y"], p["z"])


def path_length(tele):
    return sum(math.dist(xyz(a), xyz(b)) for a, b in zip(tele, tele[1:]))


def distance_to_box(p, box):
    """3D distance from point p to an oriented box {"center": [x, y, z], "half_extents": [hx, hy, hz], "yaw_deg": a}
    (rotated about z only); 0 inside the box."""
    a = math.radians(box.get("yaw_deg", 0.0))
    dx, dy, dz = (p[i] - box["center"][i] for i in range(3))
    lx = dx * math.cos(a) + dy * math.sin(a)
    ly = -dx * math.sin(a) + dy * math.cos(a)
    h = box["half_extents"]
    ex, ey, ez = (max(abs(v) - hv, 0.0) for v, hv in zip((lx, ly, dz), h))
    return math.sqrt(ex * ex + ey * ey + ez * ez)


def target_side(start_xy, start_yaw_deg, target_xy, ahead_deg=30.0, behind_deg=150.0):
    """Where the target is as seen from the start pose: ahead / left / right / behind, and the signed bearing from the
    start heading (+ = left, counter-clockwise like the ENU yaw)."""
    b = math.degrees(math.atan2(target_xy[1] - start_xy[1], target_xy[0] - start_xy[0]))
    rel = (b - start_yaw_deg + 180.0) % 360.0 - 180.0
    if abs(rel) <= ahead_deg:
        return "ahead", rel
    if abs(rel) >= behind_deg:
        return "behind", rel
    return ("left" if rel > 0 else "right"), rel


def swept_angle_deg(tele, center_xy, r_band=None):
    """Signed angle (deg, + = counter-clockwise seen from above) the drone swept around center_xy, counting only frames whose
    horizontal distance to the centre is inside r_band = (r_min, r_max) when given."""
    total, prev = 0.0, None
    for r in tele:
        x, y = r["pose"]["x"], r["pose"]["y"]
        d = math.hypot(x - center_xy[0], y - center_xy[1])
        if r_band and not (r_band[0] <= d <= r_band[1]):
            prev = None
            continue
        a = math.atan2(y - center_xy[1], x - center_xy[0])
        if prev is not None:
            total += math.degrees(math.atan2(math.sin(a - prev), math.cos(a - prev)))
        prev = a
    return total
