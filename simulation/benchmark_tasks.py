"""Benchmark tasks for the new Forest scenes easy_v2_101 .. easy_v2_200 (100 scenes, one task each, never used by the dataset).

Four Easy task types, 25 each (scene idx -> type cycles inspect, orbit, flyover, land):

  inspect   fly to a target and stop D = 1..10 m from its surface (3D, to the target's box), facing it
  orbit     circle a target once (clockwise / counter-clockwise seen from above) about R m from its centre, facing it, then stop
  flyover   fly over a target and stop right above it, camera looking down at it
  land      fly to a structure (house / barn / lookout tower / cairn) and land on the ground D = 5..10 m from it

Every distance in a success rule has the same tolerance, TOL_M = 5 m. Instructions name the target's side as seen from the
start heading (left / right) and never use compass words. The flight is pre-roll (take off, turn to the start heading;
not published) -> EPISODE (published, evaluated) -> post-roll (fly to a pad and land; not published; none for land).

plan(idx, env, attempt) is called by simulation/flight_runner.py inside Isaac Sim (env from scene_loader.boot, so the layout
is the one actually built, and the target box is measured on the USD stage). attempt > 0 gives another variant of the same
scene (other target / direction / distance), used when an expert flight of the previous variant was not clean.
"""
import math

import numpy as np

import camera_geometry as geo

TYPES = ("inspect", "orbit", "flyover", "land")
FIRST_IDX, N_SCENES = 101, 100
TOL_M = 5.0
START_AGL_M = 25.0
SETTLE_S = 3.0
STOP_HOLD_S = 2.0
START_TURN_DEG = (35.0, 100.0)     # the target is never in view at the start: 35..100 deg off the start heading
PRE, EPISODE, POST = "pre-roll", "episode", "post-roll"
FLYOVER_GIMBAL_DEG = 80.0           # the camera's steepest pitch; the expert stops where the target is mid-frame
LAND_RADIUS_M = 5.5                 # vegetation-free pad clearing: landing-gear contact allowed inside it
CAMERA_AHEAD_DEG = 12.0             # gimbal pitch while travelling: the camera looks where the drone flies
SIDE_TEXT = {"left": "on your left", "right": "on your right"}
TURN_TEXT = {"left": "Turn left", "right": "Turn right"}


def is_bench_scene(idx):
    return FIRST_IDX <= idx < FIRST_IDX + N_SCENES


def task_type(idx):
    return TYPES[(idx - FIRST_IDX) % len(TYPES)]


# ---- targets ------------------------------------------------------------------------------------------------------------
def _animals_words(g):
    import easy_v2
    return easy_v2._animal_words(g["animal"], g["n_placed"], g["coat"], g["mixed"])


def describe(g, focus):
    """(long, short) description of a target, without any compass / river-side words."""
    if g["type"] == "homestead":
        storey = "two-storey " if g["storeys"] == 2 else ""
        house = f"{storey}log cabin" if g["wall"] == "log" else f"{g['wall']} {storey}house".replace("red brick", "red-brick")
        return f"the {house} with the {g['roof']} roof (a garage and a parked car next to it)", f"the {house}"
    if g["type"] == "farm" and focus == "barn":
        return "the red barn with the grain silo beside it", "the red barn"
    if g["type"] == "farm":
        return f"the fenced paddock with {_animals_words(g)} next to the red barn", "the paddock"
    flock = "flock" if g["animal"] == "sheep" else "herd"
    return f"the {_animals_words(g)} grazing in an open meadow", f"the {flock}"


def candidates(lay):
    """[(group, focus, objects of the target)] with a description unique in this scene."""
    out = []
    for g in lay["groups"]:
        if g["type"] == "homestead":
            out.append((g, "house", [g["objects"][0]]))
        elif g["type"] == "farm":
            out.append((g, "barn", [g["objects"][0]]))
            an = [o for o in g["objects"] if o.get("kind") == g["animal"]]
            if an:
                out.append((g, "paddock", an))
        else:
            an = [o for o in g["objects"] if not o.get("world")]
            if an:
                out.append((g, "herd", an))
    longs = [describe(g, f)[0] for g, f, _ in out]
    return [c for c, lg in zip(out, longs) if longs.count(lg) == 1]


def _prim_path(g, o):
    return f"/World/Props/G{g['gid']}_{g['type']}/{o['name']}"


def target_box(stage, g, objs, ground):
    """Oriented box of the target, measured on the USD stage (one object: its own box and yaw; several animals: the
    axis-aligned box around all of them). Without a stage (offline preview): from the layout circles and heights."""
    if stage is not None:
        from pxr import Gf, Usd, UsdGeom
        cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render])
        prims = [stage.GetPrimAtPath(_prim_path(g, o)) for o in objs]
        if all(p and p.IsValid() for p in prims):
            if len(prims) == 1:
                b = cache.ComputeWorldBound(prims[0])
                rng_, m = b.GetRange(), b.GetMatrix()
                mid = m.Transform(rng_.GetMidpoint())
                sx = Gf.Vec3d(m[0][0], m[0][1], m[0][2]).GetLength()
                sy = Gf.Vec3d(m[1][0], m[1][1], m[1][2]).GetLength()
                sz = Gf.Vec3d(m[2][0], m[2][1], m[2][2]).GetLength()
                size = rng_.GetSize()
                return {"center": [float(mid[0]), float(mid[1]), float(mid[2])],
                        "half_extents": [float(size[0] * sx / 2), float(size[1] * sy / 2), float(size[2] * sz / 2)],
                        "yaw_deg": float(math.degrees(math.atan2(m[0][1], m[0][0]))), "source": "USD world bound"}
            lo, hi = [1e9] * 3, [-1e9] * 3
            for p in prims:
                r = cache.ComputeWorldBound(p).ComputeAlignedRange()
                lo = [min(lo[i], r.GetMin()[i]) for i in range(3)]
                hi = [max(hi[i], r.GetMax()[i]) for i in range(3)]
            return {"center": [(lo[i] + hi[i]) / 2 for i in range(3)], "half_extents": [(hi[i] - lo[i]) / 2 for i in range(3)],
                    "yaw_deg": 0.0, "source": "USD world bounds (axis-aligned, all animals)"}
    xs = [o["xy"][0] for o in objs]
    ys = [o["xy"][1] for o in objs]
    r = max(max(o["r"], 0.5) for o in objs)
    z0 = min(ground(o["xy"][0], o["xy"][1]) for o in objs)
    top = max(o["top_z"] for o in objs)
    return {"center": [(min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2, (z0 + top) / 2],
            "half_extents": [(max(xs) - min(xs)) / 2 + r * 0.7, (max(ys) - min(ys)) / 2 + r * 0.7, (top - z0) / 2],
            "yaw_deg": 0.0, "source": "layout circles (offline preview only)"}


_WORLD = {}


def world_obstacles(stage):
    """Static obstacles of the Forest valley that are not props, as ((x, y, r, top_z) list, (x, y, r) bushes): every
    instanced tree / rock / cliff / boulder and every landmark, measured on the USD stage
    (collision/collision_shapes.static_obstacles), and the ground-cover bushes. Without a stage (offline preview): the same
    list saved by jobs/audit_coverage.sbatch in environments/Forest/scene/obstacle_sample/world_obstacles.json, no bushes."""
    key = id(stage)
    if key not in _WORLD:
        if stage is not None:
            import collision_shapes
            allo = collision_shapes.static_obstacles(stage, include_groundcover=True)
            _WORLD[key] = ([o[:4] for o in allo if "Groundcover" not in o[4]],
                           [o[:3] for o in allo if "Groundcover" in o[4]])
        else:
            import json
            from pathlib import Path
            f = Path(__file__).resolve().parents[1] / "environments/Forest/scene/obstacle_sample/world_obstacles.json"
            _WORLD[key] = ([tuple(o[:4]) for o in json.loads(f.read_text())], [])
    return _WORLD[key]


_CUR_WORLD = []          # set by plan() for the scene being planned
_CUR_SHRUBS = []         # ground-cover bushes (x, y, r): not solid, but the expert never lands in one


def _obstacles(lay, skip):
    """Every built object except the target, and every static obstacle of the valley: (x, y, r, top_z)."""
    return [(o["xy"][0], o["xy"][1], max(o["r"], 0.5), o["top_z"]) for g in lay["groups"] for o in g["objects"]
            if not o.get("world") and id(o) not in skip] + list(_CUR_WORLD)


ROUTE_MARGIN_M = 3.0      # the whole planned flight keeps this far (horizontally, or above the top) from every obstacle
ROUTE_MARGIN_LOW_M = 1.0  # ... except on the vertical take-off / landing inside a pad clearing


def route_clear(beats, start_xy, yaw_deg, ground, obst, skip_xy=()):
    """Fly the beats with the offline pilot model and check every point against the obstacles. -> (ok, why)."""
    import pilot
    fr, p = pilot.simulate_offline(beats, ground, start_xy, math.radians(yaw_deg))
    if not p.done or getattr(p, "failed", False):
        return False, "offline pilot did not finish"
    if not obst:
        return True, None
    O = np.array(obst, dtype=float)
    for f in fr[::3]:
        low = f["beat"].startswith(("take off", "land")) or f["beat"] in ("takeoff", "land")
        m = ROUTE_MARGIN_LOW_M if low else ROUTE_MARGIN_M
        d = np.hypot(O[:, 0] - f["x"], O[:, 1] - f["y"])
        hit = (d < O[:, 2] + m) & (f["z"] < O[:, 3] + m)
        if hit.any():
            i = int(np.argmax(hit))
            return False, f"route passes {d[i] - O[i, 2]:.1f} m from an obstacle (top {O[i, 3]:.1f}) at beat '{f['beat']}'"
    return True, None


# ---- A* transit (opt-in, simulation/astar_route.py) --------------------------------------------------------------------
# BENCH_ROUTE_PLANNER=astar: the transit legs (start -> target, target -> post-roll pad) are planned with A* around the
# obstacles instead of one straight line; BENCH_ROUTE_AGL=<m> sets their preferred height above the ground (default: the
# straight route's own height). Unset (= "direct"): the straight route, exactly as the Forest Easy batch of 2026-11 was made.
def route_planner():
    import os
    return os.environ.get("BENCH_ROUTE_PLANNER", "direct"), os.environ.get("BENCH_ROUTE_AGL")


def _astar_transit(pre, ep, post, start_xy, yaw_deg, ground, obst, agl_env):
    """Replace the first goto of the episode and of the post-roll by A* legs. -> (ep, post) or (None, why)."""
    import astar_route
    import pilot

    def legs(frm, b):
        agl = float(agl_env) if agl_env else b.get("agl", START_AGL_M)
        goal = (b["xy"][0], b["xy"][1], ground(*b["xy"]) + agl)
        w = astar_route.plan(obst, ground, frm, goal, agl)
        return None if w is None else astar_route.legs_to_beats(w, b, b.get("label", "fly"))

    s0 = (start_xy[0], start_xy[1], ground(*start_xy) + START_AGL_M)   # where the pre-roll leaves the drone
    first = legs(s0, ep[0])
    if first is None:
        return None, "A*: no route to the target"
    ep = first + ep[1:]
    if post:
        fr, p = pilot.simulate_offline(pre + ep, ground, start_xy, math.radians(yaw_deg))
        if not p.done or getattr(p, "failed", False):
            return None, "A*: offline pilot did not finish the episode"
        back = legs((fr[-1]["x"], fr[-1]["y"], fr[-1]["z"]), post[0])
        if back is None:
            return None, "A*: no route to the post-roll pad"
        post = back + post[1:]
    return ep, post


def _clear_of(obst, x, y, z, margin):
    return all(math.hypot(x - ox, y - oy) > r + margin or z > top + margin for ox, oy, r, top in obst)


def _segment_clear(obst, a, b, z, margin, n=20):
    return all(_clear_of(obst, a[0] + (b[0] - a[0]) * k / n, a[1] + (b[1] - a[1]) * k / n, z, margin) for k in range(n + 1))


# ---- instructions -------------------------------------------------------------------------------------------------------
def instructions(typ, side, long, short, rng, D=None, R=None, sense=None, plural=False):
    """Canonical instruction, 3 paraphrases and a short form; every one names the side and keeps the number."""
    s, turn = SIDE_TEXT[side], TURN_TEXT[side]
    it = "them" if plural else "it"
    if typ == "inspect":
        canon = f"The target is {s}. Fly to {long} and stop about {D} m from {it}, facing {it}."
        para = [f"{turn} and head for {long}; come to a halt roughly {D} m away from {it} with the camera pointed at {it}.",
                f"Look for {long} off to the {side}, then approach {it} and hold position about {D} m from {it} while facing {it}.",
                f"Your goal is on the {side}: get within about {D} m of {short}, keep {it} in view and stop there."]
        sh = f"Stop about {D} m from {short} {s}, facing {it}."
    elif typ == "orbit":
        its = "their" if plural else "its"
        # Since 2026-11-24 (the user's decision) the drone circles FACING the target, camera on it the whole way, so the
        # direction is only said as clockwise / counter-clockwise seen from above ("keeping it on your right" described a
        # nose-first circle and is no longer true). Two wordings of the direction, picked per episode.
        clock = f"once {sense} (seen from above)"
        alt = f"once in a {sense} direction as seen from above"
        a, b = (clock, alt) if rng.random() < 0.5 else (alt, clock)
        canon = (f"The target is {s}. Fly to {long} and circle {it} {a}, facing {it} all the way round, about {R} m from "
                 f"{its} centre, then stop.")
        para = [f"{turn}, find {long} and make one full {sense} loop around {it} (viewed from above), keeping {it} in front "
                f"of the camera the whole way, roughly {R} m from {its} centre, then hold position.",
                f"Off to the {side} is {short}: go around {it} a single time, {sense} as seen from above, keeping about {R} m "
                f"from {its} centre with the camera on {it}, and stop when the circle is complete.",
                f"Your goal is on the {side}: orbit {short} {b} while looking at {it}, at a radius of about {R} m, then halt."]
        sh = f"Circle {short} {s} {a}, facing {it}, about {R} m from {its} centre, then stop."
    elif typ == "flyover":
        its = "their" if plural else "its"
        canon = f"The target is {s}. Fly over {long} and stop right above {its} centre, with the camera looking down at {it}."
        para = [f"{turn}, head for {long} and hover directly over the middle of {it}, camera aimed down at {it}.",
                f"Off to the {side} is {short}: get above {its} centre and stop there, looking straight down on {it}.",
                f"Your goal is on the {side}: position yourself over the centre of {short} and stop, keeping {it} in view below you."]
        sh = f"Stop right above the centre of {short} {s}, looking down at {it}."
    else:
        canon = f"The target is {s}. Fly to {long} and land about {D} m from {it}."
        para = [f"{turn}, head for {long} and touch down on the ground roughly {D} m away from {it}.",
                f"Off to the {side} is {short}: go there and set the drone down about {D} m from {it}.",
                f"Your goal is on the {side}: land on the ground at a distance of about {D} m from {short}."]
        sh = f"Land about {D} m from {short} {s}."
    return {"instruction": canon, "paraphrases": para, "short": sh,
            "source": "templates in simulation/benchmark_tasks.py (side from the start heading, no compass words)"}


# ---- plan ---------------------------------------------------------------------------------------------------------------
def _pads():
    from missions import PADS
    return {k: v[0] for k, v in PADS.items()}


def _start(rng, P, txy, near=(45.0, 190.0), exclude=()):
    keys = sorted(P, key=lambda k: math.hypot(P[k][0] - txy[0], P[k][1] - txy[1]))
    ok = [k for k in keys if near[0] < math.hypot(P[k][0] - txy[0], P[k][1] - txy[1]) < near[1] and k not in exclude]
    if not ok:
        return None
    return str(rng.choice(ok[:4]))


def _pre_roll(rng, start_xy, txy, ground):
    """Take off from the start pad and settle on a heading 35..100 deg away from the target (target not in view)."""
    sx, sy = start_xy
    bearing = math.atan2(txy[1] - sy, txy[0] - sx)
    turn = math.radians(float(rng.uniform(*START_TURN_DEG))) * (1 if rng.random() < 0.5 else -1)
    head = math.atan2(math.sin(bearing + turn), math.cos(bearing + turn))
    g0 = ground(sx, sy)
    ahead = (sx + 400.0 * math.cos(head), sy + 400.0 * math.sin(head), g0 + START_AGL_M - 400.0 * math.tan(math.radians(10.0)))
    beats = [dict(type="takeoff", agl=START_AGL_M, label="take off", phase=PRE),
             dict(type="hover", s=SETTLE_S, look_at=ahead, align_first=True, label="settle at the start heading", phase=PRE)]
    return beats, head, turn


def _post_roll(P, end_xy, exclude=None):
    k = min((k for k in P if k != exclude), key=lambda k: math.hypot(P[k][0] - end_xy[0], P[k][1] - end_xy[1]))
    return [dict(type="goto", xy=P[k], agl=22.0, speed=0.85, radius=4.0, label=f"fly to {k} pad", phase=POST),
            dict(type="land", xy=P[k], label=f"land {k}", pad=k, phase=POST)], k


def _inspect(rng, lay, g, objs, box, ground, start_xy):
    c = box["center"]
    top = c[2] + box["half_extents"][2]
    obst = _obstacles(lay, {id(o) for o in objs})
    base = math.atan2(start_xy[1] - c[1], start_xy[0] - c[0])
    for k in range(24):
        a = base + float(rng.uniform(-0.6, 0.6)) if k == 0 else base + (k // 2 + 1) * 0.35 * (1 if k % 2 else -1)
        ux, uy = math.cos(a), math.sin(a)
        far = (c[0] + 40 * ux, c[1] + 40 * uy)
        z = max(top - 1.0, ground(*far) + 2.5, c[2])
        gap = geo.distance_to_box((c[0], c[1], z), box)   # > 0 only when z is above the box (small targets)
        d_min = max(1, int(math.ceil(gap + 0.5)))
        if d_min > 10:
            continue
        D = int(rng.integers(d_min, 11))
        lo, hi = 0.0, 80.0
        for _ in range(60):                                # horizontal offset along the ray with box distance D
            mid = (lo + hi) / 2
            if geo.distance_to_box((c[0] + mid * ux, c[1] + mid * uy, z), box) < D:
                lo = mid
            else:
                hi = mid
        s = hi
        stand = (c[0] + s * ux, c[1] + s * uy)
        z = max(z, ground(*stand) + 2.5)
        for back in (12.0, 9.0, 7.0):
            app = (c[0] + (s + back) * ux, c[1] + (s + back) * uy)
            if math.hypot(app[0] - g["xy"][0], app[1] - g["xy"][1]) + 2.0 > g.get("clear_r", 1e9):
                continue
            if not (_clear_of(obst, *app, z, 3.0) and _segment_clear(obst, app, stand, z, 2.0)):
                continue
            look = (c[0], c[1], c[2])
            beats = [dict(type="goto", xy=app, agl=START_AGL_M, speed=0.8, radius=3.0, gimbal=CAMERA_AHEAD_DEG, label="fly to the target"),
                     dict(type="hover", s=1.5, label="stop before descending",
                          look_at=(app[0] - 400.0 * ux, app[1] - 400.0 * uy,
                                   ground(*app) + START_AGL_M - 400.0 * math.tan(math.radians(CAMERA_AHEAD_DEG)))),
                     dict(type="goto", xy=app, z=z, speed=0.4, radius=1.0, look_at=look, label="descend in the clearing"),
                     dict(type="goto", xy=stand, z=z, speed=0.25, radius=0.4, look_at=look, label=f"stop {D} m from it"),
                     dict(type="hover", s=STOP_HOLD_S, look_at=look, align_first=True, label="face it and stop")]
            return beats, {"standoff_m": float(D), "expert_standoff_xyz": [stand[0], stand[1], z],
                           "expert_distance_to_box_m": round(geo.distance_to_box((stand[0], stand[1], z), box), 3)}
    return None


def _orbit(rng, lay, g, objs, box, ground, start_xy):
    c = box["center"]
    h = box["half_extents"]
    top = c[2] + h[2]
    half_xy = math.hypot(h[0], h[1])
    R = int(round(half_xy + float(rng.uniform(8.0, 13.0))))
    ring = max(ground(c[0] + R * math.cos(a), c[1] + R * math.sin(a)) for a in np.linspace(0, 2 * math.pi, 36))
    z = max(top + float(rng.uniform(7.0, 12.0)), ring + 16.0, ground(c[0], c[1]) + 16.0)
    obst = _obstacles(lay, {id(o) for o in objs})
    if not all(_clear_of(obst, c[0] + R * math.cos(a), c[1] + R * math.sin(a), z, 3.0) for a in np.linspace(0, 2 * math.pi, 36)):
        return None
    d = int(rng.choice([1, -1]))
    a0 = math.atan2(start_xy[1] - c[1], start_xy[0] - c[0])
    look = (c[0], c[1], c[2])
    beats = [dict(type="goto", xy=(c[0] + R * math.cos(a0), c[1] + R * math.sin(a0)), agl=START_AGL_M, speed=0.8, radius=3.0,
                  gimbal=CAMERA_AHEAD_DEG, label="fly to the target"),
             # facing the target the whole circle (nose + camera on its centre, sliding sideways); the user's decision of
             # 2026-11-24 -- the previous nose-first circle (face_travel=True) kept the target out of the image ~94 % of it
             dict(type="orbit", center=look, radius=float(R), z=z, laps=1.0, dir=d, speed=4.0, face_travel=False,
                  label="circle it once (facing it, camera on it)"),
             dict(type="hover", s=STOP_HOLD_S, look_at=look, align_first=True, label="face it and stop")]
    return beats, {"radius_m": float(R), "direction": "counter-clockwise" if d > 0 else "clockwise", "orbit_z": z}


def _flyover(rng, lay, g, objs, box, ground, start_xy):
    c = box["center"]
    top = c[2] + box["half_extents"][2]
    h_above = float(rng.uniform(14.0, 20.0))
    z = top + h_above
    a = math.atan2(start_xy[1] - c[1], start_xy[0] - c[0])
    off = (z - c[2]) * math.tan(math.radians(90.0 - FLYOVER_GIMBAL_DEG))   # stop short so the target is mid-frame
    stop = (c[0] + off * math.cos(a), c[1] + off * math.sin(a))
    look = (c[0], c[1], c[2])
    ux, uy = -math.cos(a), -math.sin(a)                                     # direction of travel (start -> target)
    ahead = (stop[0] + 400.0 * ux, stop[1] + 400.0 * uy,
             ground(*stop) + START_AGL_M - 400.0 * math.tan(math.radians(CAMERA_AHEAD_DEG)))
    beats = [dict(type="goto", xy=stop, agl=START_AGL_M, speed=0.6, radius=3.0, gimbal=CAMERA_AHEAD_DEG, label="fly over the target"),
             dict(type="hover", s=1.5, look_at=ahead, label="stop, camera still ahead"),   # brake before tilting down
             dict(type="goto", xy=stop, z=z, speed=0.3, radius=0.6, gimbal=FLYOVER_GIMBAL_DEG, label="settle above it"),
             dict(type="hover", s=STOP_HOLD_S, look_at=look, label="look down at it and stop")]
    return beats, {"height_above_top_m": round(h_above, 2), "expert_stop_xyz": [stop[0], stop[1], z]}


LANDMARK_TARGETS = {          # landmarks that are unique in every scene: name -> (prim, long, short, offline box)
    "tower": ("/World/Landmarks/Lookout", "the fire lookout tower (a wooden cabin on tall legs)", "the lookout tower",
              {"center": [130.0, 85.0, 6.0], "half_extents": [2.0, 2.0, 6.0], "yaw_deg": 0.0}),
    "cairn": ("/World/Landmarks/Cairn", "the stone cairn (a pile of stacked stones)", "the stone cairn",
              {"center": [180.0, -55.0, 1.0], "half_extents": [1.5, 1.5, 1.0], "yaw_deg": 0.0}),
}


def _landmark_box(stage, prim, offline, ground):
    if stage is not None:
        from pxr import Usd, UsdGeom
        p = stage.GetPrimAtPath(prim)
        if p and p.IsValid():
            r = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render]) \
                .ComputeWorldBound(p).ComputeAlignedRange()
            lo, hi = r.GetMin(), r.GetMax()
            return {"center": [(lo[i] + hi[i]) / 2 for i in range(3)], "half_extents": [(hi[i] - lo[i]) / 2 for i in range(3)],
                    "yaw_deg": 0.0, "source": "USD world bound (axis-aligned)"}
    b = dict(offline, source="approximate (offline preview only)")
    g = ground(*b["center"][:2])
    b["center"] = [b["center"][0], b["center"][1], g + b["half_extents"][2]]
    return b


def structures(lay, stage, ground):
    """[(name, long, short, box, layout objects)] of the structures a land task can name: every house and barn with a
    unique description, the fire lookout tower and the stone cairn."""
    out = []
    for g, focus, objs in candidates(lay):
        if focus in ("house", "barn"):
            long, short = describe(g, focus)
            out.append((f"{focus}_{g['gid']}", long, short, target_box(stage, g, objs, ground), objs))
    for name, (prim, long, short, off) in LANDMARK_TARGETS.items():
        out.append((name, long, short, _landmark_box(stage, prim, off, ground), []))
    return out


LAND_D_M = (5, 10)          # land this far from the structure (3D to its box), integer metres
LAND_SPOT_CLEAR_M = 3.0     # the expert's touchdown spot: nothing (tree, rock, prop) within this
LAND_FLAT_M = 0.35          # ... and the ground within 1.5 m of it varies by less than this


def _land(rng, lay, box, objs, ground, start_xy):
    c = box["center"]
    obst = [o for o in _obstacles(lay, {id(o) for o in objs})
            if math.hypot(o[0] - c[0], o[1] - c[1]) > 0.5]          # the landmark target itself is not an obstacle here
    D = int(rng.integers(LAND_D_M[0], LAND_D_M[1] + 1))
    base = math.atan2(start_xy[1] - c[1], start_xy[0] - c[0])
    for k in range(24):
        a = base + (k // 2) * 0.27 * (1 if k % 2 else -1)
        ux, uy = math.cos(a), math.sin(a)
        lo, hi = 0.0, 80.0
        for _ in range(50):                                     # spot on the ray whose ground-level box distance is D
            m = (lo + hi) / 2
            px, py = c[0] + m * ux, c[1] + m * uy
            if geo.distance_to_box((px, py, ground(px, py) + 0.25), box) < D:
                lo = m
            else:
                hi = m
        spot = (c[0] + hi * ux, c[1] + hi * uy)
        g0 = ground(*spot)
        ring = [ground(spot[0] + 1.5 * math.cos(t), spot[1] + 1.5 * math.sin(t)) for t in np.linspace(0, 2 * math.pi, 8)]
        if max(ring) - min(ring) > LAND_FLAT_M:
            continue
        if not _clear_of(obst, spot[0], spot[1], g0 + 0.5, LAND_SPOT_CLEAR_M):
            continue
        if any(math.hypot(spot[0] - x, spot[1] - y) < r + 1.0 for x, y, r in _CUR_SHRUBS):
            continue                                            # never set down in a bush
        beats = [dict(type="goto", xy=spot, agl=20.0, speed=0.8, radius=3.0, gimbal=CAMERA_AHEAD_DEG,
                      label="fly to the target"),
                 dict(type="land", xy=spot, pad=None, label=f"land {D} m from it")]
        half = math.hypot(box["half_extents"][0], box["half_extents"][1])
        return beats, {"standoff_m": float(D), "expert_landing_xy": [spot[0], spot[1]],
                       "expert_distance_to_box_m": round(geo.distance_to_box((spot[0], spot[1], g0 + 0.25), box), 2),
                       "landing_radius_m": round(half + D + TOL_M + 1.0, 1)}
    return None


def plan(idx, env, attempt=0):
    """-> mission dict for flight_runner.py: beats with phases, landing spec, instruction texts, bench block."""
    if not is_bench_scene(idx):
        raise ValueError(f"easy_v2_{idx:03d} is not a benchmark scene ({FIRST_IDX}..{FIRST_IDX + N_SCENES - 1})")
    typ = task_type(idx)
    lay, ground = env.layout, env.ground
    stage = getattr(getattr(env, "bridge", None), "stage", None)
    P = _pads()
    import os
    # BENCH_TEST_IGNORE_OBSTACLES=1: regression test of the collision model only (the planner does not avoid the valley's
    # static obstacles, so a route through a cliff can be flown and PhysX must report it); never set for the benchmark
    blind = os.environ.get("BENCH_TEST_IGNORE_OBSTACLES") == "1"
    _CUR_WORLD[:], _CUR_SHRUBS[:] = ([], []) if blind else world_obstacles(stage)
    rejected = {}
    for trial in range(attempt * 40, attempt * 40 + 40):
        rng = np.random.default_rng(55_001 * idx + 101 * trial + 7)
        landing = None
        if typ == "land":
            structs = structures(lay, stage, ground)
            name, long, short, box, objs = structs[int(rng.integers(len(structs)))]
            txy = (box["center"][0], box["center"][1])
            start = _start(rng, P, txy)
            if start is None:
                continue
            got = _land(rng, lay, box, objs, ground, P[start])
            if got is None:
                continue
            ep, params = got
            pre, head, turn = _pre_roll(rng, P[start], txy, ground)
            kind = name.split("_")[0]
            target = {"name": name, "kind": kind, "description": long, "xy": [float(txy[0]), float(txy[1])],
                      "point_xyz": [float(v) for v in box["center"]], "box": box,
                      "top_z": box["center"][2] + box["half_extents"][2]}
            post = []
            landing = {"pad": None, "xy": [float(txy[0]), float(txy[1])], "radius_m": params["landing_radius_m"],
                       "surface_colliders": ["/World/Terrain"], "anywhere_in_episode": True,
                       "note": "the landing gear may touch the terrain within radius_m of the target's centre (the target's "
                               "standoff ring and more) at any time of the episode; touching the target or anything else is "
                               "prohibited"}
        else:
            cand = candidates(lay)
            if not cand:
                raise ValueError(f"easy_v2_{idx:03d}: no target with a unique description")
            g, focus, objs = cand[int(rng.integers(len(cand)))]
            box = target_box(stage, g, objs, ground)
            txy = (box["center"][0], box["center"][1])
            start = _start(rng, P, txy)
            if start is None:
                continue
            fn = {"inspect": _inspect, "orbit": _orbit, "flyover": _flyover}[typ]
            got = fn(rng, lay, g, objs, box, ground, P[start])
            if got is None:
                continue
            ep, params = got
            long, short = describe(g, focus)
            pre, head, turn = _pre_roll(rng, P[start], txy, ground)
            target = {"name": f"{focus}_{g['gid']}", "kind": focus, "description": long, "xy": [float(txy[0]), float(txy[1])],
                      "point_xyz": [float(v) for v in box["center"]], "box": box,
                      "top_z": box["center"][2] + box["half_extents"][2]}
        side, rel = geo.target_side(P[start], math.degrees(head), txy)
        if side not in SIDE_TEXT:
            continue
        ins = instructions(typ, side, long, short, rng, D=int(params.get("standoff_m", 0)) or None,
                           R=int(params.get("radius_m", 0)) or None, sense=params.get("direction"),
                           plural=typ != "land" and target["kind"] == "herd")
        ep = [dict(b, phase=EPISODE) for b in ep]
        if typ != "land":
            end = ep[-2]["xy"] if "xy" in ep[-2] else txy
            post, land_pad = _post_roll(P, end)
        else:
            land_pad = None                       # land: the episode ends on the ground, no post-roll
        skip = {id(o) for o in objs}
        planner, route_agl = route_planner()
        if planner == "astar" and not blind:
            ep2, post2 = _astar_transit(pre, ep, post, P[start], math.degrees(head), ground, _obstacles(lay, skip), route_agl)
            if ep2 is None:
                rejected[trial] = post2
                continue
            ep, post = ep2, post2
        beats = pre + ep + post
        ok_route, why = (True, None) if blind else route_clear(beats, P[start], math.degrees(head), ground, _obstacles(lay, skip))
        if not ok_route:
            rejected[trial] = why
            continue
        i_ep, i_stop = len(pre), len(pre) + len(ep) - 1
        m = dict(id=f"vlncol_forest_{idx:03d}", difficulty="Easy", start=start, start_xy=P[start],
                 yaw_deg=round(math.degrees(head), 2), beats=beats, instruction=ins["instruction"], task_type=typ,
                 target_description=long, land_pad=land_pad, landing=landing)
        m["tasks"] = [{"task": typ, "target": target["name"]}]       # one task per episode
        m["bench"] = {
            "task_type": typ, "attempt": attempt, "trial": trial,
            "episode_beats": [i_ep, i_stop],
            "start_heading_deg": round(math.degrees(head), 2), "start_turn_from_target_deg": round(math.degrees(turn), 1),
            "side_at_start": side, "bearing_from_start_heading_deg": round(rel, 1),
            "route_check": {"margin_m": ROUTE_MARGIN_M, "margin_takeoff_landing_m": ROUTE_MARGIN_LOW_M,
                            "obstacles": len(_obstacles(lay, skip)), "trials_rejected": rejected},
            "camera": f"{CAMERA_AHEAD_DEG:.0f} deg down along the direction of travel while moving; on the target only when "
                      "the task asks to face / look down at it",
            "target": target, "params": params, "tolerance_m": TOL_M, "instructions": ins,
            "scene": {"environment": "Forest", "scene_mission": f"easy_v2_{idx:03d}", "source_simulation": None,
                      "start_pad": start, "landing_pad_post_roll": None if typ == "land" else land_pad},
        }
        if planner == "astar":                                      # only then: direct-route goal.json files stay as they were
            m["bench"]["route_planner"] = {"planner": "astar", "preferred_agl_m": float(route_agl) if route_agl else None,
                                           "transit_legs": sum(1 for b in ep if b["type"] == "goto" and "(leg " in b.get("label", "")) + 1}
        return m
    raise ValueError(f"easy_v2_{idx:03d} attempt {attempt}: no feasible {typ} task ({len(rejected)} routes blocked, e.g. "
                     f"{next(iter(rejected.values()), None)})")
