"""Community action brief. Every number comes from tool output (geometry +
scores); this module only arranges it. No pollution-reduction percentages are
ever stated -- no validated exposure model is wired in."""
from __future__ import annotations

from .designs import LABELS
from .model import Site
from .scoring import GOAL_LABELS


def image_prompt(site: Site, rec: dict) -> str:
    """Prompt for an optional image model. Grounded in the measured geometry so
    the rendering can't invent a layout that doesn't fit."""
    m, p = rec["metrics"], rec["params"]
    bits = [f"Conceptual street-level rendering of a redesigned urban intersection ({rec['label']}).",
            f"Street widths: {site.pave_x:.1f} m and {site.pave_y:.1f} m curb-to-curb; keep the layout consistent with the attached top-down plan.",
            f"Include about {m['green_m2']:.0f} m² of planting, {m['protected_bike_m']:.0f} m of protected bike path, "
            f"and pedestrian crossings no wider than {m['max_ped_exposure_m']:.1f} m of roadway."]
    if "icd_m" in p:
        bits.append(f"Roundabout inscribed circle diameter {p['icd_m']} m with a planted central island.")
    bits.append("Style: daytime, people walking and cycling, mature street trees, no brand names, no text.")
    return " ".join(bits)


def build_brief(site: Site, scan_row: dict | None, result: dict, goals_used: dict) -> str:
    ranked, infeasible, base = result["ranked"], result["infeasible"], result["baseline"]
    L = []
    L.append(f"# Community Action Brief: {site.name}")
    L.append("")
    L.append("> **Concept for discussion only.** Not a construction design or a traffic study. "
             "Measurements are simplified and must be verified in the field; costs and calming effects are relative placeholders.")
    L.append("")
    if site.data_source and "SYNTHETIC" in site.data_source.upper():
        L.append("> ⚠️ This site uses **synthetic placeholder data**, not a real intersection.")
        L.append("")
    n = iter(range(1, 20))
    L.append(f"## {next(n)}. Why this intersection")
    if scan_row and scan_row.get("score") is not None:
        top = sorted(scan_row["contributions"].items(), key=lambda kv: -kv[1])[:3]
        from .scoring import SCAN_LABELS
        L.append(f"Opportunity score **{scan_row['score']}/100** (a prompt to investigate, not a safety rating). "
                 "Largest contributors: " + "; ".join(f"{SCAN_LABELS[k]} ({v})" for k, v in top) + ".")
        if scan_row["missing_indicators"]:
            L.append(f"*Missing data (excluded, not guessed):* {', '.join(scan_row['missing_indicators'])}. "
                     f"Confidence: {scan_row['confidence']}.")
    else:
        L.append("No scan score available (site entered manually).")
    L.append(f"Today the widest crossing exposes pedestrians to **{base['max_ped_exposure_m']} m** of roadway, "
             f"with **{base['green_m2']:.0f} m²** of planting and **{base['protected_bike_m']:.0f} m** of protected bike path in the study area.")
    L.append("")
    L.append(f"## {next(n)}. What the agent recommends")
    if not ranked:
        L.append("**No option passed the feasibility screen.** See blockers below.")
    else:
        r = ranked[0]
        m = r["metrics"]
        L.append(f"Top option under the chosen priorities: **{r['label']}** (score {r['score']}/100).")
        L.append("")
        L.append("| Metric (±40 m study window) | Existing | " + r["label"] + " |")
        L.append("|---|---|---|")
        L.append(f"| Max pedestrian roadway exposure (m) | {base['max_ped_exposure_m']} | {m['max_ped_exposure_m']} |")
        L.append(f"| Planting (m²) | {base['green_m2']:.0f} | {m['green_m2']:.0f} |")
        L.append(f"| Protected bike path (m) | {base['protected_bike_m']:.0f} | {m['protected_bike_m']:.0f} |")
        L.append(f"| Roadway pavement change (m²) | - | {m['pavement_change_m2']:+.0f} |")
        L.append(f"| Traffic calming (ordinal 1-3, qualitative) | - | {m['traffic_calming_ordinal']} |")
        L.append(f"| Relative cost index (1-5, NOT dollars) | - | {m['relative_cost_index']} |")
        L.append("")
        if len(ranked) > 1:
            L.append("**All feasible options:** " + "; ".join(f"{x['label']} ({x['score']})" for x in ranked) + ".")
    L.append("")
    L.append("Priorities used: " + ", ".join(f"{GOAL_LABELS[k]} {round(v * 100)}%" for k, v in goals_used.items()) +
             ". Changing them can change the ranking.")
    L.append("")
    if infeasible:
        L.append(f"## {next(n)}. Options screened out")
        for i in infeasible:
            L.append(f"- **{i['label']}**: " + " ".join(i["blockers"]))
        L.append("")
    if ranked:
        r = ranked[0]
        L.append(f"## {next(n)}. Review flags before going further")
        for w in r["warnings"]:
            L.append(f"- {w}")
        L.append("")
        L.append(f"## {next(n)}. Assumptions behind the numbers")
        for a in r["assumptions"]:
            L.append(f"- {a}")
        L.append("- Air-pollution benefits are **not quantified**. Plausible pathways (calmer traffic, more walking/cycling, "
                 "planting) need a validated exposure model and local traffic data to estimate.")
        L.append("")
        L.append(f"## {next(n)}. Suggested next steps")
        L += ["1. Verify the measured widths against aerial imagery and a field visit.",
              "2. Gather traffic counts, truck volumes and collision details for this location.",
              "3. Share with residents, nearby schools and businesses; record what they want changed.",
              "4. Ask a licensed traffic engineer to review turning paths, drainage and ADA access.",
              "5. Check cost estimates against current City of Oakland / Caltrans unit costs."]
        L.append("")
        L.append("## Optional: prompt for an image model (grounded in the measured plan)")
        L.append("```")
        L.append(image_prompt(site, r))
        L.append("```")
    return "\n".join(L) + "\n"
