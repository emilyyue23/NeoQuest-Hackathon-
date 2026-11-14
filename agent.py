"""Agent layer.

Two interchangeable runners:

* ``run_bedrock_agent``: a Strands Agents SDK agent on Amazon Bedrock. The LLM
  decides which tools to call and writes the explanation; every number comes
  from tool output. Needs AWS credentials + Bedrock model access.
* ``run_offline_agent``: a deterministic planner that follows the same tool
  sequence with no LLM. Used for the credential-free demo and for evals.

Both return the same structure: {"trace": [...], "summary": str, "artifacts": {...}}.
"""
from __future__ import annotations

import json
import os

from . import tools as T

SYSTEM_PROMPT = """You are EcoEquity Street Agent, an urban-design assistant that helps residents and planners
explore safer, greener, more equitable street redesigns for specific intersections.

Rules you must follow:
1. NEVER state a number (areas, lengths, widths, scores, costs) that did not come from a tool result. Do the
   arithmetic with tools, not in your head.
2. Typical workflow: scan_intersections -> pick a site (or use the one the user names) -> evaluate_redesign ->
   propose_redesign for the best option(s) -> generate_brief. Call only the tools you need.
3. Do not default to a roundabout. Compare all options and explain why options were screened out.
4. Never claim a specific reduction in air pollution or emissions; no validated exposure model is available.
   You may describe plausible pathways and clearly label them as uncertain.
5. Distinguish facts (tool outputs), assumptions (listed in tool results) and your own interpretation.
6. Always surface warnings from tool results (freight routes, emergency access, transit, ADA, drainage) and say
   this is a concept for discussion, not an engineering design.
7. If indicators are missing for a site, say so; do not guess them.
Keep answers concise: recommendation, key measured numbers, trade-offs, flags, next steps."""


# ---------------------------------------------------------------------------
# Offline (deterministic) agent
# ---------------------------------------------------------------------------
def _step(trace: list, tool: str, args: dict, fn, summarise):
    out = fn(**args)
    trace.append({"tool": tool, "args": args, "result_summary": summarise(out)})
    return out


def run_offline_agent(request: str = "Find the best intersection to redesign and recommend a layout.",
                      site_id: str | None = None, goals: dict | None = None, scan_weights: dict | None = None) -> dict:
    trace: list[dict] = []
    goals_json = json.dumps(goals) if goals else ""
    scan = _step(trace, "scan_intersections",
                 {"weights_json": json.dumps(scan_weights) if scan_weights else "", "top_n": 5}, T.scan_intersections,
                 lambda o: f"{o['total_candidates']} sites scanned; top: " +
                           ", ".join(f"{c['site_id']} ({c['score']})" for c in o["candidates"][:3]))
    chosen = site_id or scan["candidates"][0]["site_id"]
    ev = _step(trace, "evaluate_redesign", {"site_id": chosen, "goals_json": goals_json}, T.evaluate_redesign,
               lambda o: "ranked: " + (", ".join(f"{r['label']} ({r['score']})" for r in o["ranked"]) or "none feasible") +
                         (f"; screened out: {', '.join(i['label'] for i in o['infeasible'])}" if o["infeasible"] else ""))
    sketch = None
    if ev["ranked"]:
        best = ev["ranked"][0]["design"]
        sketch = _step(trace, "propose_redesign", {"site_id": chosen, "design_type": best}, T.propose_redesign,
                       lambda o: f"{o['label']} sketch saved; green {o['metrics']['green_m2']} m², "
                                 f"bike {o['metrics']['protected_bike_m']} m, exposure {o['metrics']['max_ped_exposure_m']} m")
    brief = _step(trace, "generate_brief", {"site_id": chosen, "goals_json": goals_json}, T.generate_brief,
                  lambda o: f"brief written to {o['brief_path']}")

    site = T.get_site(chosen)
    if ev["ranked"]:
        r = ev["ranked"][0]
        m, b = r["metrics"], ev["baseline"]
        summary = (f"**{site.name}** ranks first of {scan['total_candidates']} scanned sites. "
                   f"Best option under your priorities: **{r['label']}** (score {r['score']}/100). "
                   f"It would cut the widest pedestrian crossing from {b['max_ped_exposure_m']} m to "
                   f"{m['max_ped_exposure_m']} m of roadway, add {m['green_m2']:.0f} m² of planting and "
                   f"{m['protected_bike_m']:.0f} m of protected bike path within the study window.")
        if ev["infeasible"]:
            summary += " Screened out: " + "; ".join(f"{i['label']} ({i['blockers'][0]})" for i in ev["infeasible"]) + "."
        summary += " This is a concept for discussion; flags and assumptions are in the brief."
    else:
        summary = f"No redesign option passed the feasibility screen for **{site.name}**. See blockers in the brief."
    return {"mode": "offline", "request": request, "site_id": chosen, "trace": trace, "summary": summary,
            "artifacts": {"comparison_png": ev["comparison_path"], "sketch_png": sketch["sketch_path"] if sketch else None,
                          "brief_md": brief["brief_path"]}, "evaluation": ev, "scan": scan}


# ---------------------------------------------------------------------------
# Bedrock / Strands agent
# ---------------------------------------------------------------------------
def bedrock_available() -> tuple[bool, str]:
    try:
        import strands  # noqa: F401
    except Exception as e:  # pragma: no cover
        return False, f"strands-agents not installed ({e})"
    try:
        import boto3
        if boto3.Session().get_credentials() is None:
            return False, "no AWS credentials found"
    except Exception as e:  # pragma: no cover
        return False, f"boto3 problem ({e})"
    return True, "ok"


def build_strands_agent(model_id: str | None = None, region: str | None = None):
    from strands import Agent, tool
    from strands.models import BedrockModel

    model_id = model_id or os.environ.get("ECOEQUITY_MODEL_ID")
    kwargs = {}
    if model_id:
        kwargs["model_id"] = model_id
    region = region or os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION")
    if region:
        kwargs["region_name"] = region
    model = BedrockModel(**kwargs) if kwargs else BedrockModel()
    return Agent(model=model, system_prompt=SYSTEM_PROMPT, tools=[tool(f) for f in T.TOOLS])


def _trace_from_messages(messages: list) -> list[dict]:
    trace = []
    for msg in messages:
        for block in msg.get("content", []) if isinstance(msg.get("content"), list) else []:
            if "toolUse" in block:
                tu = block["toolUse"]
                trace.append({"tool": tu.get("name"), "args": tu.get("input"), "result_summary": "(see tool result)"})
    return trace


def run_bedrock_agent(request: str, model_id: str | None = None) -> dict:
    agent = build_strands_agent(model_id)
    result = agent(request)
    return {"mode": "bedrock", "request": request, "trace": _trace_from_messages(agent.messages),
            "summary": str(result), "artifacts": {}}


def run_agent(request: str, mode: str = "auto", **kw) -> dict:
    """mode: 'offline' | 'bedrock' | 'auto'.

    Bedrock is strictly opt-in: ``mode='bedrock'`` or env ``ECOEQUITY_USE_BEDROCK=1``
    (so ambient AWS credentials are never used by surprise). If an opted-in
    Bedrock run fails in 'auto' mode we fall back to the offline planner and say so.
    """
    want_bedrock = mode == "bedrock" or (mode == "auto" and os.environ.get("ECOEQUITY_USE_BEDROCK") == "1")
    note = None
    if want_bedrock:
        ok, why = bedrock_available()
        if ok:
            try:
                return run_bedrock_agent(request, kw.get("model_id"))
            except Exception as e:
                if mode == "bedrock":
                    raise
                note = f"Bedrock call failed ({type(e).__name__}); used offline planner instead."
        elif mode == "bedrock":
            raise RuntimeError(f"Bedrock mode unavailable: {why}")
        else:
            note = f"Bedrock unavailable ({why}); used offline planner instead."
    out = run_offline_agent(request, site_id=kw.get("site_id"), goals=kw.get("goals"),
                            scan_weights=kw.get("scan_weights"))
    if note:
        out["note"] = note
    return out
