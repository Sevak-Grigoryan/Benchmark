"""A* route planner for the expert's transit legs (opt-in; added 2026-11-24, after the Forest Easy batch was started).

The default planner flies a straight line from the start to the target and rejects the variant if an obstacle is in the
way. With BENCH_ROUTE_PLANNER=astar (simulation/benchmark_tasks.py) the transit legs are planned here instead, so the
expert flies AROUND trees / rocks / cliffs / buildings, the way OpenFly (arXiv 2502.18041) plans on a voxel map:

  1. occupancy grid   The obstacles are already known exactly (benchmark_tasks._obstacles: every built object and every
                      static obstacle of the valley as vertical cylinders (x, y, r, top_z), measured on the USD stage),
                      so no point cloud is needed. For every XY cell of CELL_M the grid stores the lowest safe height:
                      max(ground + MIN_AGL_M, top + margin of every cylinder within r + margin). A 3D voxel (cell, z
                      layer) is free when its z is at or above that height. (Cylinders have no overhangs, so this 2.5D
                      height map is exactly the 3D voxel map.)
  2. A*               26-connected search over (cell, layer). Cost = path length + CLIMB_COST per metre climbed, and
                      every metre flown costs HIGH_COST more per 10 m above the preferred height (ground + agl), so the
                      expert goes around an obstacle when that is cheaper than climbing over it. Heuristic: straight-line
                      distance (admissible: every cost is >= the length).
  3. simplification   Line-of-sight pruning: keep only the turning points; a leg is accepted only if it is clear at the
                      LOWER of its two end heights (the pilot does not climb in a straight 3D line). Legs shorter than
                      MIN_LEG_M (one-layer height steps) are merged into the next leg at the higher height.
  4. beats            Every leg becomes a `goto` beat with an explicit z, flown closed-loop by
                      environments/common/sim/pilot.py through PX4 exactly like the straight route (camera ahead).

The planned route is still checked afterwards by benchmark_tasks.route_clear (offline pilot flight vs. the obstacles), the
coverage gate and the camera-ahead rule, and PhysX still judges every contact in the real flight.
Pure Python + numpy, no Isaac Sim:  python3 simulation/astar_route.py  -> self-test.
"""
import heapq
import math

import numpy as np

CELL_M = 2.0          # horizontal grid resolution
LAYER_M = 2.0         # vertical grid resolution
MIN_AGL_M = 5.0       # never plan lower than this above the ground
MARGIN_M = 4.5        # keep-out around every obstacle (route_clear checks 3 m; +1.5 m for grid and flight error)
PAD_M = 80.0          # the search area is the start/goal box grown by this
CLIMB_COST = 1.0      # extra cost per metre climbed
HIGH_COST = 0.6       # flying 10 m above the preferred height costs 60 % more per metre
MAX_ABOVE_M = 40.0    # highest layer: preferred height + this
MAX_EXPANSIONS = 600_000


class Grid:
    """Lowest safe height per XY cell over the area that holds the points `pts` (x, y) plus PAD_M."""

    def __init__(self, obst, ground, pts, margin=MARGIN_M, cell=CELL_M, pad=PAD_M):
        P = np.array(pts, dtype=float)
        self.x0, self.y0 = P[:, 0].min() - pad, P[:, 1].min() - pad
        self.cell = cell
        self.nx = int(math.ceil((P[:, 0].max() + pad - self.x0) / cell)) + 1
        self.ny = int(math.ceil((P[:, 1].max() + pad - self.y0) / cell)) + 1
        xs = self.x0 + cell * np.arange(self.nx)
        ys = self.y0 + cell * np.arange(self.ny)
        # ground: highest of the cell centre and its 4 half-cell corners (a cell is as high as its highest part)
        g = np.full((self.nx, self.ny), -1e9)
        for ox, oy in ((0, 0), (-.5, -.5), (.5, -.5), (-.5, .5), (.5, .5)):
            gg = np.array([[ground(x + ox * cell, y + oy * cell) for y in ys] for x in xs])
            g = np.maximum(g, gg)
        self.ground = g
        floor = g + MIN_AGL_M
        half = cell * math.sqrt(0.5)                            # a cell is blocked if any part of it is inside the keep-out
        for ox, oy, r, top in obst:
            rr = r + margin + half
            i0, i1 = max(0, int((ox - rr - self.x0) / cell)), min(self.nx - 1, int((ox + rr - self.x0) / cell) + 1)
            j0, j1 = max(0, int((oy - rr - self.y0) / cell)), min(self.ny - 1, int((oy + rr - self.y0) / cell) + 1)
            if i0 > i1 or j0 > j1:
                continue
            X, Y = np.meshgrid(xs[i0:i1 + 1], ys[j0:j1 + 1], indexing="ij")
            hit = np.hypot(X - ox, Y - oy) < rr
            sub = floor[i0:i1 + 1, j0:j1 + 1]
            sub[hit] = np.maximum(sub[hit], top + margin)
        self.safe_z = floor

    def ij(self, x, y):
        return (min(self.nx - 1, max(0, int(round((x - self.x0) / self.cell)))),
                min(self.ny - 1, max(0, int(round((y - self.y0) / self.cell)))))

    def xy(self, i, j):
        return float(self.x0 + i * self.cell), float(self.y0 + j * self.cell)

    def safe_at(self, x, y):
        return float(self.safe_z[self.ij(x, y)])

    def segment_clear(self, a, b, step=1.0):
        """Straight leg a -> b (x, y, z) clear at the lower of its two end heights."""
        z = min(a[2], b[2])
        n = max(1, int(math.hypot(b[0] - a[0], b[1] - a[1]) / step))
        return all(self.safe_at(a[0] + (b[0] - a[0]) * k / n, a[1] + (b[1] - a[1]) * k / n) <= z + 1e-6 for k in range(n + 1))


def plan(obst, ground, start, goal, agl, margin=MARGIN_M):
    """A* from start (x, y, z) to goal (x, y, z). -> list of waypoints (x, y, z) after `start`, ending at the goal's xy
    (at the goal's z, or higher if the goal itself is inside the keep-out), or None if no route exists."""
    G = Grid(obst, ground, [start[:2], goal[:2]], margin)
    z_lo = float(G.ground.min() + MIN_AGL_M)
    pref = G.ground + agl                                       # preferred height per cell
    z_hi = float(pref.max() + MAX_ABOVE_M)
    nz = int(math.ceil((z_hi - z_lo) / LAYER_M)) + 1
    zs = z_lo + LAYER_M * np.arange(nz)

    def layer(z):
        return min(nz - 1, max(0, int(math.ceil((z - z_lo) / LAYER_M - 1e-9))))

    si, sj = G.ij(*start[:2])
    gi, gj = G.ij(*goal[:2])
    sk = max(layer(start[2]), layer(G.safe_z[si, sj]))
    gk = max(layer(goal[2]), layer(G.safe_z[gi, gj]))
    if zs[sk] > z_hi or zs[gk] > z_hi:
        return None
    free = zs[None, None, :] >= G.safe_z[:, :, None] - 1e-6     # (nx, ny, nz) voxel map
    above = np.maximum(0.0, zs[None, None, :] - pref[:, :, None])
    c, L = G.cell, LAYER_M
    nbrs = [(di, dj, dk, math.sqrt((di * c) ** 2 + (dj * c) ** 2 + (dk * L) ** 2) + CLIMB_COST * max(dk, 0) * L)
            for di in (-1, 0, 1) for dj in (-1, 0, 1) for dk in (-1, 0, 1) if (di, dj, dk) != (0, 0, 0)]
    gx, gy, gz = G.xy(gi, gj) + (zs[gk],)

    def h(i, j, k):
        x, y = G.xy(i, j)
        return math.sqrt((x - gx) ** 2 + (y - gy) ** 2 + (zs[k] - gz) ** 2)

    s, goal_n = (si, sj, sk), (gi, gj, gk)
    best, parent = {s: 0.0}, {s: None}
    openq = [(h(*s), 0.0, s)]
    n_exp = 0
    while openq:
        _, gcost, u = heapq.heappop(openq)
        if u == goal_n:
            break
        if gcost > best.get(u, math.inf):
            continue
        n_exp += 1
        if n_exp > MAX_EXPANSIONS:
            return None
        i, j, k = u
        for di, dj, dk, w in nbrs:
            v = (i + di, j + dj, k + dk)
            if not (0 <= v[0] < G.nx and 0 <= v[1] < G.ny and 0 <= v[2] < nz) or not free[v]:
                continue
            if di and dj and not (free[i + di, j, k] and free[i, j + dj, k]):
                continue                                        # no corner cutting
            ng = gcost + w * (1.0 + HIGH_COST * above[v] / 10.0)
            if ng < best.get(v, math.inf):
                best[v], parent[v] = ng, u
                heapq.heappush(openq, (ng + h(*v), ng, v))
    if goal_n not in parent:
        return None
    cells = []
    u = goal_n
    while u is not None:
        cells.append(u)
        u = parent[u]
    cells.reverse()
    pts = [(start[0], start[1], max(start[2], float(zs[sk])))] + [G.xy(i, j) + (float(zs[k]),) for i, j, k in cells[1:-1]] \
        + [(goal[0], goal[1], float(zs[gk]))]
    return merge_short(G, pts[0], simplify(G, pts))


MIN_LEG_M = 8.0       # legs shorter than this (A* steps of one layer) are merged into the next one


def merge_short(G, start, w):
    """A waypoint closer than MIN_LEG_M to the previous one (a one-layer A* height step) is dropped: the next waypoint
    (or else the previous one) takes the higher of the two heights, if the changed legs are still clear. The goal's xy
    is never moved. No 2 m 'staircase' legs for the pilot to stop at."""
    out = list(w)
    pt = lambda n: start if n < 0 else out[n]  # noqa: E731
    changed = True
    while changed:
        changed = False
        for n in range(len(out) - 1):
            if math.hypot(out[n][0] - pt(n - 1)[0], out[n][1] - pt(n - 1)[1]) >= MIN_LEG_M:
                continue
            z = max(out[n][2], out[n + 1][2])
            a = (out[n + 1][0], out[n + 1][1], z)               # drop out[n], raise the next one
            if G.segment_clear(pt(n - 1), a) and (n + 2 >= len(out) or G.segment_clear(a, out[n + 2])):
                out[n:n + 2] = [a]
                changed = True
                break
            if n >= 1:                                          # drop out[n], raise the previous one
                b = (out[n - 1][0], out[n - 1][1], max(out[n - 1][2], out[n][2]))
                if G.segment_clear(pt(n - 2), b) and G.segment_clear(b, out[n + 1]):
                    out[n - 1:n + 1] = [b]
                    changed = True
                    break
    return out


def simplify(G, pts):
    """Line-of-sight pruning: from each kept point jump to the farthest point still reachable by a clear straight leg."""
    out, i = [], 0
    while i < len(pts) - 1:
        j = len(pts) - 1
        while j > i + 1 and not G.segment_clear(pts[i], pts[j]):
            j -= 1
        out.append(pts[j])
        i = j
    return out


def legs_to_beats(waypoints, last, label="fly to the target"):
    """goto beats for the waypoints; the last one keeps the original beat's speed / radius / gimbal / look_at."""
    beats = []
    for n, (x, y, z) in enumerate(waypoints):
        if n == len(waypoints) - 1:
            b = {k: v for k, v in last.items() if k not in ("agl", "follow_terrain")}
            b.update(xy=(x, y), z=z)
        else:
            b = dict(type="goto", xy=(x, y), z=z, speed=last.get("speed", 0.8), radius=4.0, label=f"{label} (leg {n + 1})")
            b.update({k: last[k] for k in ("gimbal", "phase") if k in last})
        beats.append(b)
    return beats


def _selftest():
    flat = lambda x, y: 0.0  # noqa: E731
    wall = [(50.0, float(y), 3.0, 15.0) for y in range(-40, 41, 4)]        # a row of 15 m trees across the way
    w = plan(wall, flat, (0, 0, 8), (100, 0, 8), agl=8.0)
    assert w is not None
    G = Grid(wall, flat, [(0, 0), (100, 0)])
    prev = (0, 0, 8)
    for p in w:
        assert G.segment_clear(prev, p), (prev, p)
        prev = p
    over = max(p[2] for p in w) > 15 + MARGIN_M - 1e-6
    around = max(abs(p[1]) for p in w) > 40
    print("waypoints", [tuple(round(v, 1) for v in p) for p in w], "| goes", "over" if over else "around" if around else "?")
    assert over or around
    assert plan([], flat, (0, 0, 25), (200, 50, 25), agl=25.0)[-1][:2] == (200, 50), "open field must be one leg"
    print("astar_route self-test OK")


if __name__ == "__main__":
    _selftest()
