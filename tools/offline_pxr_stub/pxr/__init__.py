"""Offline stand-in for pxr (OpenUSD), used ONLY by the tools in tools/ on a machine without Isaac Sim.

It lets the scene / planner modules that import pxr at the top load, so the plans can be previewed offline. Any real USD
call returns a dummy, so nothing that needs the stage works here: the authoritative plan is the one computed inside Isaac
Sim by the flight job. Never put this folder on the path of a flight or render job.
"""


class _Any:
    def __getattr__(self, k):
        return _Any()

    def __call__(self, *a, **k):
        return _Any()


for _n in ["Gf", "Sdf", "UsdGeom", "UsdLux", "UsdPhysics", "UsdShade", "Usd", "PhysxSchema", "Vt"]:
    globals()[_n] = _Any()
