"""Transparent scoring: candidate opportunity score for scanning sites, and
weighted multi-criteria ranking of redesign options.

Both are simple weighted sums so a judge (or resident) can see exactly why
something ranked where it did. A high scan score means "worth investigating",
NOT "proven dangerous".
"""
from __future__ import annotations

from .designs import Layout, metrics
from .model import Site

SCAN_WEIGHTS = {"P": 0.30, "B": 0.25, "G": 0.20, "E": 0.15, "C": 0.10}
SCAN_LABELS = {
    "P": "Pedestrian need (crossing width)",
    "B": "Bike-network gap",
    "G": "Green-space deficit (low canopy)",
    "E": "Environmental burden",
    "C": "Documented collisions",
}
GOAL_WEIGHTS = {"ped_safety": 0.30, "bike": 0.25, "green": 0.20, "calming": 0.15, "cost": 0.10}
GOAL_LABELS = {
    "ped_safety": "Shorter pedestrian crossings",
    "bike": "Protected bike connectivity",
    "green": "Green space",
    "calming": "Traffic calming",
    "cost": "Low relative cost",
}


def _minmax(values: dict[str, float | None]) -> dict[str, float | None]:
    present = [v for v in values.values() if v is not None]
    if not present:
        return {k: None for k in values}
    lo, hi = min(present), max(present)
    return {k: (None if v is None else (50.0 if hi == lo else 100 * (v - lo) / (hi - lo))) for k, v in values.items()}


def _normalise_weights(w: dict | None, defaults: dict) -> dict:
    w = {k: float((w or {}).get(k, defaults[k])) for k in defaults}
    s = sum(max(v, 0) for v in w.values())
    if s <= 0:
        return dict(defaults)
    return {k: max(v, 0) / s for k, v in w.items()}


def scan_sites(sites: list[Site], weights: dict | None = None) -> list[dict]:
    """Rank candidate sites by opportunity score. Missing indicators are
    excluded (weights renormalised) and flagged, never silently imputed."""
    w = _normalise_weights(weights, SCAN_WEIGHTS)
    P = _minmax({s.id: s.max_crossing_m() for s in sites})
    C = _minmax({s.id: (None if s.collisions_5yr is None else float(s.collisions_5yr)) for s in sites})
    rows = []
    for s in sites:
        comp = {
            "P": P[s.id],
            "B": s.bike_gap,
            "G": None if s.canopy_pct is None else max(0.0, min(100.0, 100.0 - s.canopy_pct)),
            "E": s.env_burden_pctile,
            "C": C[s.id],
        }
        avail = {k: v for k, v in comp.items() if v is not None}
        wsum = sum(w[k] for k in avail)
        score = sum(w[k] * v for k, v in avail.items()) / wsum if wsum else None
        missing = [SCAN_LABELS[k] for k, v in comp.items() if v is None]
        rows.append({
            "site_id": s.id, "name": s.name,
            "score": None if score is None else round(score, 1),
            "components": {k: (None if v is None else round(v, 1)) for k, v in comp.items()},
            "contributions": {k: round(w[k] / wsum * v, 1) for k, v in avail.items()} if wsum else {},
            "missing_indicators": missing,
            "confidence": "high" if not missing else ("medium" if len(missing) == 1 else "low"),
            "data_source": s.data_source,
            "flags": [f for f, on in (("freight route", s.freight_route), ("transit stop", s.transit_stop),
                                      ("emergency route", s.emergency_route)) if on],
        })
    rows.sort(key=lambda r: (-1 if r["score"] is None else -r["score"], r["site_id"]))
    return rows


def rank_designs(layouts: dict[str, Layout], goals: dict | None = None) -> dict:
    """Rank feasible designs by weighted score. Infeasible designs are listed
    with their blockers instead of being scored."""
    w = _normalise_weights(goals, GOAL_WEIGHTS)
    base = layouts["existing"]
    base_m = metrics(base, base)
    designs = [l for k, l in layouts.items() if k != "existing"]
    feas = [l for l in designs if l.feasible]
    ms = {l.key: metrics(l, base) for l in designs}
    max_bike = max([ms[l.key]["protected_bike_m"] for l in feas] or [0])
    max_green = max([ms[l.key]["green_m2"] for l in feas] or [0])
    ranked = []
    for l in feas:
        m = ms[l.key]
        crit = {
            "ped_safety": max(0.0, (base_m["max_ped_exposure_m"] - m["max_ped_exposure_m"]) / base_m["max_ped_exposure_m"]),
            "bike": 0.0 if max_bike == 0 else m["protected_bike_m"] / max_bike,
            "green": 0.0 if max_green == 0 else m["green_m2"] / max_green,
            "calming": m["traffic_calming_ordinal"] / 3.0,
            "cost": 1.0 - (m["relative_cost_index"] - 1) / 4.0,
        }
        contrib = {k: round(100 * w[k] * crit[k], 1) for k in crit}
        ranked.append({"design": l.key, "label": l.label, "score": round(sum(contrib.values()), 1),
                       "criteria": {k: round(v, 2) for k, v in crit.items()}, "contributions": contrib,
                       "metrics": m, "warnings": l.warnings, "assumptions": l.assumptions, "params": l.params})
    ranked.sort(key=lambda r: (-r["score"], r["design"]))
    for i, r in enumerate(ranked, 1):
        r["rank"] = i
    infeasible = [{"design": l.key, "label": l.label, "blockers": l.blockers} for l in designs if not l.feasible]
    return {"weights": {k: round(v, 3) for k, v in w.items()}, "baseline": base_m, "ranked": ranked,
            "infeasible": infeasible}
