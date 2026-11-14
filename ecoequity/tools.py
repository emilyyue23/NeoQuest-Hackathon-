"""Callable tools for the agent. Plain Python functions returning JSON-safe
dicts; all geometry and numbers are computed here, never by the LLM.

`scan_intersections`, `propose_redesign`, `evaluate_redesign` and
`generate_brief` are the tool surface. Strands wraps them in agent.py.
"""
from __future__ import annotations

import json
import os
from dataclasses import asdict
from pathlib import Path

from .brief import build_brief
from .designs import DESIGN_KEYS, LABELS, build_all, metrics
from .model import Site, load_sites
from .render import render_comparison, render_single
from .scoring import GOAL_WEIGHTS, SCAN_WEIGHTS, _normalise_weights, rank_designs, scan_sites

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = Path(os.environ.get("ECOEQUITY_OUT", ROOT / "output"))
_SITES: dict[str, Site] = {}


def load_default_sites(path: str | Path | None = None) -> list[Site]:
    for s in load_sites(path or ROOT / "data" / "sample_sites.csv"):
        _SITES[s.id] = s
    return list(_SITES.values())


def register_site(site: Site) -> Site:
    problems = site.validate()
    if problems:
        raise ValueError("; ".join(problems))
    _SITES[site.id] = site
    return site


def get_site(site_id: str) -> Site:
    if not _SITES:
        load_default_sites()
    if site_id not in _SITES:
        raise KeyError(f"Unknown site_id '{site_id}'. Known: {sorted(_SITES)}")
    return _SITES[site_id]


def all_sites() -> list[Site]:
    if not _SITES:
        load_default_sites()
    return list(_SITES.values())


def _weights(raw: str | dict | None) -> dict | None:
    if not raw:
        return None
    return json.loads(raw) if isinstance(raw, str) else raw


# ---------------------------------------------------------------------------
def scan_intersections(weights_json: str = "", top_n: int = 5) -> dict:
    """Scan candidate intersections and rank them by an opportunity score.

    The score combines pedestrian crossing width, bike-network gap, low tree
    canopy, environmental burden and documented collisions. A high score means
    "worth investigating", not "proven dangerous". Missing indicators are
    excluded and flagged rather than guessed.

    Args:
        weights_json: Optional JSON object with any of the keys P, B, G, E, C
            (pedestrian, bike gap, green deficit, environmental burden, collisions).
        top_n: How many top candidates to return.

    Returns:
        Dict with the weights used and a ranked list of candidates including
        per-indicator evidence and data-quality flags.
    """
    w = _normalise_weights(_weights(weights_json), SCAN_WEIGHTS)
    rows = scan_sites(all_sites(), w)
    return {"weights_used": {k: round(v, 3) for k, v in w.items()}, "candidates": rows[: int(top_n)],
            "total_candidates": len(rows)}


def propose_redesign(site_id: str, design_type: str) -> dict:
    """Build a conceptual, geometry-based redesign of one intersection and
    save a before/after top-down sketch.

    Args:
        site_id: Id of a scanned intersection (e.g. "S01").
        design_type: One of roundabout, protected_intersection,
            curb_extensions, traffic_circle.

    Returns:
        Measured metrics (planting area, protected bike length, pedestrian
        crossing exposure, pavement change), feasibility blockers, warnings,
        assumptions and the path of the saved sketch.
    """
    if design_type not in DESIGN_KEYS:
        raise ValueError(f"design_type must be one of {DESIGN_KEYS}")
    site = get_site(site_id)
    layouts = build_all(site)
    lay, base = layouts[design_type], layouts["existing"]
    path = render_single(site, lay, base, OUT_DIR / f"{site_id}_{design_type}.png")
    return {"site_id": site_id, "design": design_type, "label": LABELS[design_type], "feasible": lay.feasible,
            "blockers": lay.blockers, "warnings": lay.warnings, "assumptions": lay.assumptions,
            "params": lay.params, "metrics": metrics(lay, base), "baseline": metrics(base, base), "sketch_path": path}


def evaluate_redesign(site_id: str, goals_json: str = "") -> dict:
    """Compare ALL redesign options for a site against the existing layout and
    rank the feasible ones for the user's priorities.

    Args:
        site_id: Id of a scanned intersection.
        goals_json: Optional JSON object with any of the keys ped_safety, bike,
            green, calming, cost giving relative importance (any positive numbers).

    Returns:
        Weights used, baseline metrics, ranked options with per-criterion
        contributions, and options screened out with their blockers.
    """
    site = get_site(site_id)
    layouts = build_all(site)
    res = rank_designs(layouts, _weights(goals_json))
    res["site_id"] = site_id
    res["comparison_path"] = render_comparison(site, layouts, OUT_DIR / f"{site_id}_comparison.png")
    return res


def generate_brief(site_id: str, goals_json: str = "") -> dict:
    """Write a one-page community action brief (markdown) for a site, built only
    from computed results, with explicit assumptions and review flags.

    Args:
        site_id: Id of a scanned intersection.
        goals_json: Optional priorities, as for evaluate_redesign.

    Returns:
        Path of the saved markdown brief and its text.
    """
    site = get_site(site_id)
    layouts = build_all(site)
    res = rank_designs(layouts, _weights(goals_json))
    scan_row = next((r for r in scan_sites(all_sites()) if r["site_id"] == site_id), None)
    text = build_brief(site, scan_row, res, res["weights"])
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / f"{site_id}_brief.md"
    path.write_text(text, encoding="utf-8")
    return {"site_id": site_id, "brief_path": str(path), "brief_markdown": text}


TOOLS = [scan_intersections, propose_redesign, evaluate_redesign, generate_brief]
