"""Evaluation panel: checks that the agent's tools respect geometry, budget-style
constraints, ranking logic, honesty about data and the 'no invented pollution
numbers' rule. Run `python -m ecoequity.evals`."""
from __future__ import annotations

import json
import re
import tempfile
from dataclasses import replace

from shapely.ops import unary_union

from . import tools as T
from .brief import build_brief
from .designs import DESIGN_KEYS, build_all, metrics
from .model import Site
from .scoring import rank_designs, scan_sites


def _site(**kw) -> Site:
    base = dict(id="T", name="Test site", row_x=20, row_y=20, pave_x=12, pave_y=12, parking_x=2.4, parking_y=2.4,
                canopy_pct=10, env_burden_pctile=80, collisions_5yr=3, bike_gap=70)
    base.update(kw)
    return Site(**base)


def run_evals() -> list[dict]:
    res: list[dict] = []

    def check(name, ok, detail=""):
        res.append({"check": name, "passed": bool(ok), "detail": detail})

    sites = T.all_sites()

    # 1. Geometry stays inside the right-of-way and layers don't overlap
    bad = []
    for s in sites:
        for k, lay in build_all(s).items():
            if not lay.feasible:
                continue
            tol = 1e-6
            for name in ("pavement", "green", "bike", "hardscape"):
                g = getattr(lay, name)
                if not g.is_empty and g.difference(lay.row.buffer(tol)).area > 1e-3:
                    bad.append(f"{s.id}/{k}/{name} leaves ROW")
            if lay.green.intersection(lay.pavement).area > 1e-3:
                bad.append(f"{s.id}/{k} green overlaps pavement")
            if lay.bike.intersection(lay.pavement).area > 1e-3:
                bad.append(f"{s.id}/{k} bike overlaps pavement")
    check("Geometry stays inside right-of-way; green/bike never overlap roadway", not bad, "; ".join(bad[:3]))

    # 2. Narrow streets: roundabout + protected lanes must be rejected, with reasons
    narrow = build_all(_site(row_x=15, row_y=15, pave_x=9, pave_y=9))
    check("Narrow street: roundabout rejected with reason", not narrow["roundabout"].feasible and narrow["roundabout"].blockers)
    check("Narrow street: protected lanes rejected with reason",
          not narrow["protected_intersection"].feasible and narrow["protected_intersection"].blockers)

    # 3. Roundabout is not the default: wide-but-tight site shouldn't always pick it
    tops = set()
    for s in sites:
        r = rank_designs(build_all(s))
        if r["ranked"]:
            tops.add(r["ranked"][0]["design"])
    check("Agent does not default to one design across sites", len(tops) > 1, f"top designs seen: {sorted(tops)}")

    # 4. Constraint flags
    fr = build_all(_site(freight_route=True))
    check("Freight route flagged on roundabout", any("Freight" in w or "freight" in w for w in fr["roundabout"].warnings))
    art = build_all(_site(class_x="arterial"))
    check("Arterial street: traffic circle rejected", not art["traffic_circle"].feasible)
    check("Emergency + transit flags surfaced",
          any("Emergency" in w for w in build_all(_site(emergency_route=True))["curb_extensions"].warnings) and
          any("Transit" in w for w in build_all(_site(transit_stop=True))["curb_extensions"].warnings))

    # 5. Ranking responds to weights in the expected direction
    L = build_all(_site(row_x=24, row_y=24, pave_x=14, pave_y=14))
    g_only = rank_designs(L, {"ped_safety": 0, "bike": 0, "green": 1, "calming": 0, "cost": 0})["ranked"]
    max_green = max(g_only, key=lambda r: r["metrics"]["green_m2"])["design"]
    check("Green-only weights pick the design with most planting", g_only[0]["design"] == max_green)
    b_only = rank_designs(L, {"ped_safety": 0, "bike": 1, "green": 0, "calming": 0, "cost": 0})["ranked"]
    max_bike = max(b_only, key=lambda r: r["metrics"]["protected_bike_m"])["design"]
    check("Bike-only weights pick the design with most protected bike path", b_only[0]["design"] == max_bike)
    c_only = rank_designs(L, {"ped_safety": 0, "bike": 0, "green": 0, "calming": 0, "cost": 1})["ranked"]
    check("Cost-only weights pick the cheapest feasible design",
          c_only[0]["metrics"]["relative_cost_index"] == min(r["metrics"]["relative_cost_index"] for r in c_only))
    inf_only = rank_designs(build_all(_site(row_x=15, row_y=15, pave_x=9, pave_y=9)))
    check("Infeasible designs never appear in the ranking",
          all(r["design"] not in {i["design"] for i in inf_only["infeasible"]} for r in inf_only["ranked"]))

    # 6. Honesty about data: missing indicators flagged, not imputed
    row = scan_sites([_site(canopy_pct=None, collisions_5yr=None)])[0]
    check("Missing indicators flagged and confidence lowered",
          len(row["missing_indicators"]) == 2 and row["confidence"] != "high" and row["components"]["G"] is None)

    # 7. Numbers in tool output are reproducible (determinism)
    a = json.dumps(rank_designs(build_all(sites[0])), sort_keys=True)
    b = json.dumps(rank_designs(build_all(sites[0])), sort_keys=True)
    check("Results are deterministic", a == b)

    # 8. Brief honesty: no invented pollution %, caveats present
    s = sites[0]
    lays = build_all(s)
    r = rank_designs(lays)
    text = build_brief(s, scan_sites(sites)[0], r, r["weights"])
    bad_claim = re.search(r"(\d+\s*%[^.\n]{0,60}(pollution|emission|PM2\.5|diesel))|((pollution|emission|PM2\.5|diesel)[^.\n]{0,60}\d+\s*%)", text, re.I)
    check("Brief states no invented pollution/emission percentages", not bad_claim, bad_claim.group(0) if bad_claim else "")
    check("Brief includes concept-only disclaimer and next steps",
          "Concept for discussion only" in text and "licensed traffic engineer" in text)
    if r["ranked"]:
        top = r["ranked"][0]["metrics"]
        check("Brief numbers match tool metrics",
              f"| Planting (m²) | {r['baseline']['green_m2']:.0f} | {top['green_m2']:.0f} |" in text)

    # 9. Tool-level: unknown ids / bad design types fail loudly
    try:
        T.propose_redesign("S01", "skatepark")
        check("Bad design_type raises", False)
    except ValueError:
        check("Bad design_type raises", True)
    try:
        T.get_site("NOPE")
        check("Unknown site raises", False)
    except KeyError:
        check("Unknown site raises", True)

    # 10. Input validation
    check("Validation catches pavement wider than right-of-way", bool(_site(row_x=10, pave_x=12).validate()))
    return res


def main():
    import os
    with tempfile.TemporaryDirectory() as d:
        os.environ.setdefault("ECOEQUITY_OUT", d)
        res = run_evals()
    width = max(len(r["check"]) for r in res)
    for r in res:
        print(("PASS" if r["passed"] else "FAIL"), r["check"].ljust(width), r["detail"])
    n = sum(r["passed"] for r in res)
    print(f"\n{n}/{len(res)} checks passed")
    raise SystemExit(0 if n == len(res) else 1)


if __name__ == "__main__":
    main()
