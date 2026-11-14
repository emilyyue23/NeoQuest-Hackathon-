"""Geometry-based conceptual redesigns.

Each design builds shapely polygons (pavement, green, protected bike, hardscape
islands, crosswalks). All numbers reported downstream are *measured from these
polygons* -- nothing is estimated by a language model.

IMPORTANT: dimensions/thresholds below are screening heuristics, not design
standards. Verify against NACTO / FHWA / Caltrans / City of Oakland guidance
and have a licensed engineer check turning paths, drainage and ADA.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

from shapely.geometry import Point, box
from shapely.geometry.base import BaseGeometry
from shapely.ops import unary_union

from .model import LEG_LEN, Site, leg_params, pavement_geom, rot, row_geom

# ---- screening parameters (all in metres) --------------------------------
BIKE_W = 1.8             # one-way protected bike lane
SEP_W = 0.9              # separator between bike lane and motor lane
STRIP_W = BIKE_W + SEP_W
MIN_MOTOR_WIDTH = 6.0    # two 3.0 m lanes must remain on a street
MIN_BULB = 1.5           # smallest curb extension worth drawing
SIDEWALK_MIN = 2.0       # sidewalk kept around a roundabout
CIRC_W = 5.0             # circulating roadway width (single lane, compact)
APRON = 1.0              # mountable truck apron around central island
SPLIT_W = 2.0            # splitter / pedestrian refuge island width
ICD_TARGET = 30.0        # inscribed circle diameter we aim for
ICD_COMPACT_MIN = 25.0   # below this -> "mini" roundabout (mountable centre)
ICD_MIN = 18.0           # below this -> does not fit
ICD_FREIGHT_COMFORT = 35.0
TRAFFIC_CIRCLE_R = 3.0

# Qualitative / relative assumptions. PLACEHOLDERS: replace with sourced evidence
# (e.g. FHWA CMF Clearinghouse for crash effects, City of Oakland unit costs).
TRAFFIC_CALMING = {"roundabout": 3, "traffic_circle": 2, "curb_extensions": 2, "protected_intersection": 2}
COST_INDEX = {"traffic_circle": 1, "curb_extensions": 2, "protected_intersection": 4, "roundabout": 5}  # relative, NOT dollars
LABELS = {
    "existing": "Existing",
    "roundabout": "Green roundabout",
    "protected_intersection": "Protected intersection",
    "curb_extensions": "Green curb extensions",
    "traffic_circle": "Neighborhood traffic circle",
}
DESIGN_KEYS = ["roundabout", "protected_intersection", "curb_extensions", "traffic_circle"]

EMPTY = unary_union([])


@dataclass
class Layout:
    key: str
    label: str
    row: BaseGeometry
    pavement: BaseGeometry
    green: BaseGeometry = EMPTY
    bike: BaseGeometry = EMPTY          # protected bike lanes / paths
    hardscape: BaseGeometry = EMPTY     # islands, separators, aprons
    crosswalks: list = field(default_factory=list)   # [(polygon, motor_exposure_m)]
    bike_len: float = 0.0
    feasible: bool = True
    blockers: list = field(default_factory=list)
    warnings: list = field(default_factory=list)
    assumptions: list = field(default_factory=list)
    params: dict = field(default_factory=dict)

    @property
    def sidewalk(self) -> BaseGeometry:
        return self.row.difference(unary_union([self.pavement, self.bike, self.hardscape, self.green]))


# ---------------------------------------------------------------------------
def baseline(site: Site) -> Layout:
    cws = []
    for k in range(4):
        p = leg_params(site, k)
        cws.append((rot(box(p["d0"] + 1.5, -p["pave"] / 2, p["d0"] + 4.5, p["pave"] / 2), k), p["pave"]))
    return Layout("existing", LABELS["existing"], row_geom(site), pavement_geom(site), crosswalks=cws)


def _blocked(site: Site, key: str, reasons: list[str], base: Layout) -> Layout:
    lay = Layout(key, LABELS[key], base.row, base.pavement, crosswalks=list(base.crosswalks))
    lay.feasible = False
    lay.blockers = reasons
    return lay


def _planters(x0, x1, y0, y1, k, on=4.0, off=4.0):
    """Alternating planter boxes along a strip (leg-local coords), rotated to leg k."""
    boxes, x = [], x0
    while x < x1 - 0.5:
        boxes.append(rot(box(x, y0, min(x + on, x1), y1), k))
        x += on + off
    return unary_union(boxes)


def _clear_crossings(lay: Layout):
    """Keep planters out of crosswalk paths."""
    keep_clear = unary_union([p for p, _ in lay.crosswalks]).buffer(0.3)
    lay.green = lay.green.difference(keep_clear)


def _common_warnings(site: Site, lay: Layout):
    if site.freight_route:
        lay.warnings.append("Freight route: confirm design-vehicle (e.g. WB-67) turning paths before any further work.")
    if site.emergency_route:
        lay.warnings.append("Emergency-vehicle route: fire/EMS access and clear widths need review.")
    if site.transit_stop:
        lay.warnings.append("Transit stop at/near the intersection: stop location and bus turning need review.")
    lay.warnings.append("Concept only: needs professional review of turning paths, drainage, sight distance and ADA.")


# ---------------------------------------------------------------------------
# A. Green roundabout
# ---------------------------------------------------------------------------
def roundabout(site: Site, base: Layout) -> Layout:
    key = "roundabout"
    row = base.row
    r_max = min(math.hypot(site.row_x / 2, site.row_y / 2), LEG_LEN - 6.0)
    # largest circle (radius r_max) fits inside the cross-shaped right-of-way
    assert Point(0, 0).buffer(r_max - 1e-6).within(row.buffer(1e-6))

    ring_ok = False
    R = min(ICD_TARGET / 2, r_max - SIDEWALK_MIN - BIKE_W)
    if 2 * R >= ICD_COMPACT_MIN:
        ring_ok = True
    else:
        R = min(ICD_TARGET / 2, r_max - SIDEWALK_MIN)
    icd = 2 * R
    blockers = []
    if icd < ICD_MIN:
        blockers.append(f"Right-of-way only fits a {icd:.0f} m circle once a {SIDEWALK_MIN:.0f} m sidewalk is kept "
                        f"(minimum screening size {ICD_MIN:.0f} m).")
    entries = {}
    for k in range(4):
        p = leg_params(site, k)
        entry = min(4.5, (p["pave"] - SPLIT_W) / 2)
        entries[k] = entry
        if entry < 3.0:
            blockers.append(f"Leg {k}: street only {p['pave']:.1f} m wide; a {SPLIT_W:.0f} m splitter leaves "
                            f"{entry:.1f} m entry lanes (< 3.0 m).")
    if blockers:
        return _blocked(site, key, sorted(set(blockers)), base)

    circ = Point(0, 0).buffer(R, quad_segs=32)
    r_c = R - CIRC_W
    inner = Point(0, 0).buffer(r_c, quad_segs=32)
    green_island = Point(0, 0).buffer(max(r_c - APRON, 0.1), quad_segs=32)
    apron = inner.difference(green_island)
    annulus = circ.difference(inner)

    carriage, splitters = [], []
    for k in range(4):
        hw = entries[k] + SPLIT_W / 2
        carriage.append(rot(box(0, -hw, LEG_LEN, hw), k))
        splitters.append(rot(box(R, -SPLIT_W / 2, R + 10, SPLIT_W / 2), k))
    carriage_u = unary_union(carriage)
    split_u = unary_union(splitters)
    pavement = unary_union([annulus, carriage_u]).difference(inner).difference(split_u)

    ring_w = BIKE_W if ring_ok else 0.0
    x0 = R + ring_w
    greens = [green_island]
    bikes, seps = [], []
    bike_len = 0.0
    ring_len = 0.0
    if ring_ok:
        ring = Point(0, 0).buffer(R + BIKE_W, quad_segs=32).difference(circ).difference(carriage_u)
        bikes.append(ring)
        ring_len = ring.area / BIKE_W
        bike_len += ring_len
    strips = 0
    for k in range(4):
        p = leg_params(site, k)
        hw = entries[k] + SPLIT_W / 2
        e = p["pave"] / 2 - hw
        for s in (+1, -1):
            def yy(a, b):  # leg-local lateral interval, mirrored for side s
                return (min(s * a, s * b), max(s * a, s * b))
            if ring_ok and e >= STRIP_W:
                y0, y1 = yy(p["pave"] / 2 - BIKE_W, p["pave"] / 2)
                bikes.append(rot(box(x0, y0, LEG_LEN, y1), k))
                y0, y1 = yy(p["pave"] / 2 - STRIP_W, p["pave"] / 2 - BIKE_W)
                seps.append(rot(box(x0, y0, LEG_LEN, y1), k))
                greens.append(_planters(x0, LEG_LEN, y0, y1, k))
                bike_len += LEG_LEN - x0
                strips += 1
                rest = e - STRIP_W
                if rest >= 1.2:
                    ya, yb = yy(hw, hw + rest)
                    greens.append(_planters(x0, LEG_LEN, ya, yb, k))
            elif e >= 1.2:
                ya, yb = yy(hw, p["pave"] / 2)
                greens.append(_planters(x0, LEG_LEN, ya, yb, k))

    cws = []
    for k in range(4):
        p = leg_params(site, k)
        xs = R + ring_w + 5.0
        hw = entries[k] + SPLIT_W / 2
        cws.append((rot(box(xs, -hw, xs + 3.0, hw), k), entries[k]))

    lay = Layout(key, LABELS[key], row, pavement, green=unary_union(greens), bike=unary_union(bikes),
                 hardscape=unary_union([apron, split_u] + seps), crosswalks=cws, bike_len=bike_len,
                 params=dict(icd_m=round(icd, 1), central_island_green_radius_m=round(max(r_c - APRON, 0.1), 1),
                             separated_bike_ring=ring_ok, ring_len_m=round(ring_len, 1), approach_bike_strips=strips,
                             type="compact roundabout" if icd >= ICD_COMPACT_MIN else "mini-roundabout"))
    lay.assumptions += [
        f"Single-lane roundabout, {CIRC_W:.0f} m circulating roadway, {APRON:.0f} m mountable apron, {SPLIT_W:.0f} m splitter islands.",
        "Pedestrian exposure = entry lane width per crossing stage (assumes refuge in splitter island, two-stage crossing).",
        f"Planting shown at ~50% coverage along strips (4 m planters / 4 m gaps); central island fully planted inside the apron.",
    ]
    if icd < ICD_COMPACT_MIN:
        lay.warnings.append(f"Inscribed circle is only {icd:.0f} m: treated as a mini-roundabout with a mountable centre; "
                            "planting capacity is limited.")
    if not ring_ok:
        lay.warnings.append("No room for a separated bike path around the circle: bikes would share the circulating lane.")
    if site.freight_route and icd < ICD_FREIGHT_COMFORT:
        lay.warnings.append(f"Freight route with a {icd:.0f} m inscribed circle: large trucks may not be able to navigate "
                            "without relying on the apron. Turning-path analysis required.")
    if "arterial" in (site.class_x, site.class_y):
        lay.warnings.append("Arterial approach: roundabout capacity needs a traffic study (volumes not modelled here).")
    _clear_crossings(lay)
    _common_warnings(site, lay)
    return lay


# ---------------------------------------------------------------------------
# B. Protected intersection (Dutch-style)
# ---------------------------------------------------------------------------
def protected_intersection(site: Site, base: Layout) -> Layout:
    key = "protected_intersection"
    ok = {}
    blockers = []
    for k in range(4):
        p = leg_params(site, k)
        avail = (p["pave"] - MIN_MOTOR_WIDTH) / 2
        ok[p["street"]] = avail >= STRIP_W
        if avail < STRIP_W:
            blockers.append(f"Street {p['street']}: {p['pave']:.1f} m wide leaves {max(avail, 0):.1f} m per side "
                            f"(need {STRIP_W:.1f} m for bike lane + separator while keeping {MIN_MOTOR_WIDTH:.0f} m for traffic).")
    if not any(ok.values()):
        return _blocked(site, key, sorted(set(blockers)), base)

    bikes, seps, greens = [], [], []
    bike_len = 0.0
    for k in range(4):
        p = leg_params(site, k)
        if not ok[p["street"]]:
            continue
        for s in (+1, -1):
            y_lo = lambda a, b: (min(s * a, s * b), max(s * a, s * b))
            y0, y1 = y_lo(p["pave"] / 2 - BIKE_W, p["pave"] / 2)
            bikes.append(rot(box(p["d0"], y0, LEG_LEN, y1), k))
            y0, y1 = y_lo(p["pave"] / 2 - STRIP_W, p["pave"] / 2 - BIKE_W)
            seps.append(rot(box(p["d0"], y0, LEG_LEN, y1), k))
            greens.append(_planters(p["d0"], LEG_LEN, y0, y1, k))
            bike_len += LEG_LEN - p["d0"]
    islands = []
    if ok["x"] and ok["y"]:
        hx, hy = site.pave_y / 2, site.pave_x / 2
        core = box(-hx, -hy, hx, hy)
        for sx in (+1, -1):
            for sy in (+1, -1):
                islands.append(Point(sx * hx, sy * hy).buffer(3.0, quad_segs=16).intersection(core))
    bike_u, sep_u = unary_union(bikes), unary_union(seps)
    isl_u = unary_union(islands)
    pavement = base.pavement.difference(unary_union([bike_u, sep_u, isl_u]))

    exposure = {}
    cws = []
    for k in range(4):
        p = leg_params(site, k)
        exp = p["pave"] - 2 * STRIP_W if ok[p["street"]] else p["pave"]
        exposure[k] = exp
        cws.append((rot(box(p["d0"] + 1.5, -p["pave"] / 2, p["d0"] + 4.5, p["pave"] / 2), k), exp))

    lay = Layout(key, LABELS[key], base.row, pavement, green=unary_union(greens), bike=bike_u,
                 hardscape=unary_union([sep_u, isl_u]), crosswalks=cws, bike_len=bike_len,
                 params=dict(streets_converted=[s for s, v in ok.items() if v], corner_islands=len(islands)))
    lay.assumptions += [
        f"Bike lane {BIKE_W} m + separator {SEP_W} m per side, carved from existing roadway while keeping {MIN_MOTOR_WIDTH:.0f} m for motor traffic.",
        "Pedestrian exposure = roadway width left for motor traffic (bike lane crossing counted separately).",
        "Protected length counts the legs only; the conflict-point detailing inside the intersection core is not drawn or counted.",
        "Separators shown with ~50% planters; planted vs. flex-post depends on maintenance and drainage.",
    ]
    for b in sorted(set(blockers)):
        lay.warnings.append("Partial design: " + b)
    if site.parking_x or site.parking_y:
        lay.warnings.append("Converts on-street parking to bike lanes: parking loss needs community/business consultation.")
    _clear_crossings(lay)
    _common_warnings(site, lay)
    return lay


# ---------------------------------------------------------------------------
# C. Green curb extensions
# ---------------------------------------------------------------------------
def curb_extensions(site: Site, base: Layout) -> Layout:
    key = "curb_extensions"
    ext, blockers = {}, []
    for ax in "xy":
        pave, park = getattr(site, f"pave_{ax}"), getattr(site, f"parking_{ax}")
        e = min(park, (pave - MIN_MOTOR_WIDTH) / 2)
        if e < MIN_BULB:
            blockers.append(f"Street {ax}: no usable parking lane to convert "
                            f"(extension would be {max(e, 0):.1f} m; minimum {MIN_BULB} m).")
            e = 0.0
        ext[ax] = e
    if not any(ext.values()):
        return _blocked(site, key, blockers, base)
    bulbs, greens, cws = [], [], []
    for k in range(4):
        p = leg_params(site, k)
        e = ext[p["street"]]
        if e:
            for s in (+1, -1):
                a, b = sorted((s * (p["pave"] / 2 - e), s * p["pave"] / 2))
                bulb = rot(box(p["d0"], a, p["d0"] + 7.0, b), k)
                bulbs.append(bulb)
                greens.append(bulb.buffer(-0.5))
        cws.append((rot(box(p["d0"] + 1.5, -(p["pave"] / 2 - e), p["d0"] + 4.5, p["pave"] / 2 - e), k),
                    p["pave"] - 2 * e))
    bulb_u = unary_union(bulbs)
    lay = Layout(key, LABELS[key], base.row, base.pavement.difference(bulb_u), green=unary_union(greens),
                 hardscape=bulb_u.difference(unary_union(greens)), crosswalks=cws,
                 params=dict(extension_m=ext, bulb_outs=len(bulbs)))
    lay.assumptions += [
        "Bulb-outs are 7 m long and sized to the existing parking lane; no protected bike infrastructure is added.",
        "Pedestrian exposure = curb-to-curb width minus both extensions.",
        "Planting inset 0.5 m from curb edge; bioswale/stormwater function would need drainage design.",
    ]
    for b in blockers:
        lay.warnings.append("Partial design: " + b)
    _common_warnings(site, lay)
    return lay


# ---------------------------------------------------------------------------
# D. Neighborhood traffic circle
# ---------------------------------------------------------------------------
def traffic_circle(site: Site, base: Layout) -> Layout:
    key = "traffic_circle"
    blockers = []
    for ax in "xy":
        if getattr(site, f"class_{ax}") != "local":
            blockers.append(f"Street {ax} is a {getattr(site, f'class_{ax}')} street; traffic circles are only "
                            "screened as suitable for low-volume local streets.")
    if blockers:
        return _blocked(site, key, blockers, base)
    r = TRAFFIC_CIRCLE_R
    clear = min(site.pave_x, site.pave_y) / 2 - r
    isl = Point(0, 0).buffer(r, quad_segs=24)
    green = Point(0, 0).buffer(r - 0.4, quad_segs=24)
    lay = Layout(key, LABELS[key], base.row, base.pavement.difference(isl), green=green,
                 hardscape=isl.difference(green), crosswalks=list(base.crosswalks),
                 params=dict(island_radius_m=r, clearance_m=round(clear, 1)))
    lay.assumptions += [
        f"{2 * r:.0f} m diameter island; existing crossings and lane layout otherwise unchanged.",
        "No effect on crossing distance and no new bike infrastructure is assumed.",
    ]
    if clear < 3.0:
        lay.warnings.append(f"Only {clear:.1f} m of clear roadway around the island (< 3.0 m): tight for turning vehicles.")
    _common_warnings(site, lay)
    return lay


BUILDERS = {"roundabout": roundabout, "protected_intersection": protected_intersection,
            "curb_extensions": curb_extensions, "traffic_circle": traffic_circle}


def build_all(site: Site) -> dict[str, Layout]:
    base = baseline(site)
    out = {"existing": base}
    for k in DESIGN_KEYS:
        out[k] = BUILDERS[k](site, base)
    return out


def metrics(lay: Layout, base: Layout) -> dict:
    exp = [e for _, e in lay.crosswalks]
    return {
        "design": lay.key,
        "label": lay.label,
        "feasible": lay.feasible,
        "green_m2": round(lay.green.area, 1),
        "protected_bike_m": round(lay.bike_len, 1),
        "max_ped_exposure_m": round(max(exp), 1),
        "mean_ped_exposure_m": round(sum(exp) / len(exp), 1),
        "pavement_m2": round(lay.pavement.area, 1),
        "pavement_change_m2": round(lay.pavement.area - base.pavement.area, 1),
        "sidewalk_m2": round(lay.sidewalk.area, 1),
        "traffic_calming_ordinal": TRAFFIC_CALMING.get(lay.key, 0),
        "relative_cost_index": COST_INDEX.get(lay.key, 0),
    }
