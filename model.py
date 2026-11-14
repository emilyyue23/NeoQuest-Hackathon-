"""Site model: a simplified, rectilinear 4-leg intersection in metres.

Coordinates: intersection centre is (0, 0). Street "x" runs east-west,
street "y" runs north-south. Legs are numbered 0:+x, 1:+y, 2:-x, 3:-y.

Everything here is a *screening-level* abstraction. Real intersections have
curb radii, skewed legs, driveways, utilities, etc. Measurements should be
verified against aerial imagery / survey data before anyone relies on them.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass, fields
from pathlib import Path

from shapely import affinity
from shapely.geometry import box
from shapely.ops import unary_union

LEG_LEN = 40.0  # how far each leg is drawn/analysed from the centre (m)

STREET_CLASSES = ("local", "collector", "arterial")


@dataclass
class Site:
    id: str
    name: str
    # --- measurements (metres); street x = east-west, street y = north-south
    row_x: float          # right-of-way width of street x
    row_y: float
    pave_x: float         # curb-to-curb width of street x
    pave_y: float
    parking_x: float = 0.0   # width of parking lane on EACH side (0 = none)
    parking_y: float = 0.0
    bike_x: str = "none"     # none | painted
    bike_y: str = "none"
    class_x: str = "local"   # local | collector | arterial
    class_y: str = "local"
    freight_route: bool = False
    transit_stop: bool = False
    emergency_route: bool = False
    # --- indicators used by the scanner (None = missing, which is flagged)
    canopy_pct: float | None = None          # tree canopy near site, 0-100
    env_burden_pctile: float | None = None   # environmental burden percentile 0-100
    collisions_5yr: int | None = None        # documented collisions, 5 yrs
    bike_gap: float | None = None            # 0 = connected network ... 100 = no bike facilities nearby
    lat: float | None = None
    lon: float | None = None
    data_source: str = "user-entered"
    notes: str = ""

    # ---- derived helpers -------------------------------------------------
    def max_crossing_m(self) -> float:
        return max(self.pave_x, self.pave_y)

    def missing_indicators(self) -> list[str]:
        return [k for k in ("canopy_pct", "env_burden_pctile", "collisions_5yr", "bike_gap")
                if getattr(self, k) is None]

    def validate(self) -> list[str]:
        """Return a list of problems with the measurements (empty = ok)."""
        p = []
        for ax in "xy":
            row, pave = getattr(self, f"row_{ax}"), getattr(self, f"pave_{ax}")
            if pave <= 0 or row <= 0:
                p.append(f"street {ax}: widths must be positive")
            if pave > row:
                p.append(f"street {ax}: pavement ({pave} m) is wider than right-of-way ({row} m)")
            if getattr(self, f"class_{ax}") not in STREET_CLASSES:
                p.append(f"street {ax}: class must be one of {STREET_CLASSES}")
        return p


def rot(geom, k: int):
    """Rotate a leg-local geometry (leg along +x) to leg k."""
    return affinity.rotate(geom, 90 * k, origin=(0, 0))


def leg_params(site: Site, k: int) -> dict:
    """Parameters of leg k (leg-local frame: leg along +x, lateral = y)."""
    if k % 2 == 0:
        return dict(street="x", pave=site.pave_x, row=site.row_x, parking=site.parking_x,
                    bike=site.bike_x, cls=site.class_x, d0=site.pave_y / 2)
    return dict(street="y", pave=site.pave_y, row=site.row_y, parking=site.parking_y,
                bike=site.bike_y, cls=site.class_y, d0=site.pave_x / 2)


def _cross(wx: float, wy: float, length: float = LEG_LEN):
    return unary_union([box(-length, -wx / 2, length, wx / 2),
                        box(-wy / 2, -length, wy / 2, length)])


def row_geom(site: Site):
    return _cross(site.row_x, site.row_y)


def pavement_geom(site: Site):
    return _cross(site.pave_x, site.pave_y)


# ---------------------------------------------------------------------------
# CSV loading
# ---------------------------------------------------------------------------
_BOOL = {"1", "true", "yes", "y", "t"}


def _parse(field_type: str, raw: str):
    raw = (raw or "").strip()
    if field_type.startswith("float | None"):
        return float(raw) if raw else None
    if field_type.startswith("int | None"):
        return int(float(raw)) if raw else None
    if field_type == "float":
        return float(raw)
    if field_type == "bool":
        return raw.lower() in _BOOL
    return raw


def load_sites(path: str | Path) -> list[Site]:
    types = {f.name: str(f.type) for f in fields(Site)}
    sites: list[Site] = []
    with open(path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            kwargs = {}
            for k, v in row.items():
                if k not in types:
                    continue
                optional = "None" in types[k]
                if (v or "").strip() == "" and not optional:
                    continue  # fall back to the dataclass default
                kwargs[k] = _parse(types[k], v)
            sites.append(Site(**kwargs))
    return sites
