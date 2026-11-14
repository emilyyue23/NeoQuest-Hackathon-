"""EcoEquity AI - Reimagining Our Streets (Streamlit demo).

Run:  streamlit run app.py
"""
from __future__ import annotations

import json

import pandas as pd
import streamlit as st

from ecoequity import tools as T
from ecoequity.agent import bedrock_available, run_agent
from ecoequity.designs import LABELS
from ecoequity.evals import run_evals
from ecoequity.model import Site
from ecoequity.scoring import GOAL_LABELS, GOAL_WEIGHTS, SCAN_LABELS, SCAN_WEIGHTS

st.set_page_config(page_title="EcoEquity AI - Reimagining Our Streets", page_icon="🌳", layout="wide")

st.title("🌳 EcoEquity AI: Reimagining Our Streets")
st.caption("An agent that scans intersections, compares redesigns, sketches them from real geometry, "
           "and writes a community action brief. **Concept for discussion, not engineering design.**")

sites = T.all_sites()
if any("SYNTHETIC" in (s.data_source or "").upper() for s in sites):
    st.warning("Sample sites are **synthetic placeholders**. Enter a real intersection's measurements in the sidebar "
               "(from aerial imagery or a field visit) to analyse an actual location.")

# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("1 · What matters most?")
    st.caption("Scan weights decide which intersections surface; priorities decide which redesign wins.")
    with st.expander("Scan weights (find sites)", expanded=False):
        sw = {k: st.slider(SCAN_LABELS[k], 0, 100, int(v * 100), key=f"sw_{k}") for k, v in SCAN_WEIGHTS.items()}
    st.subheader("Redesign priorities")
    gw = {k: st.slider(GOAL_LABELS[k], 0, 100, int(v * 100), key=f"gw_{k}") for k, v in GOAL_WEIGHTS.items()}
    if sum(gw.values()) == 0:
        st.error("Set at least one priority above zero.")
        st.stop()

    st.header("2 · Your own intersection")
    with st.form("custom"):
        name = st.text_input("Name", "My intersection")
        c1, c2 = st.columns(2)
        row_x = c1.number_input("ROW width E-W street (m)", 8.0, 60.0, 20.0, 0.5)
        row_y = c2.number_input("ROW width N-S street (m)", 8.0, 60.0, 20.0, 0.5)
        pave_x = c1.number_input("Curb-to-curb E-W (m)", 5.0, 40.0, 12.0, 0.1)
        pave_y = c2.number_input("Curb-to-curb N-S (m)", 5.0, 40.0, 12.0, 0.1)
        park_x = c1.number_input("Parking lane E-W (m, each side)", 0.0, 4.0, 2.4, 0.1)
        park_y = c2.number_input("Parking lane N-S (m, each side)", 0.0, 4.0, 2.4, 0.1)
        cls_x = c1.selectbox("Class E-W", ["local", "collector", "arterial"])
        cls_y = c2.selectbox("Class N-S", ["local", "collector", "arterial"])
        fr = st.checkbox("Freight / truck route")
        tr = st.checkbox("Transit stop at intersection")
        em = st.checkbox("Emergency-vehicle route")
        with st.expander("Optional scan indicators"):
            canopy = st.number_input("Tree canopy nearby (%)", 0.0, 100.0, 10.0)
            burden = st.number_input("Environmental burden percentile", 0.0, 100.0, 80.0)
            coll = st.number_input("Collisions, last 5 yrs", 0, 500, 0)
            gap = st.number_input("Bike-network gap (0-100)", 0.0, 100.0, 70.0)
        go = st.form_submit_button("Add / update intersection")
    if go:
        try:
            T.register_site(Site(id="CUSTOM", name=name, row_x=row_x, row_y=row_y, pave_x=pave_x, pave_y=pave_y,
                                 parking_x=park_x, parking_y=park_y, class_x=cls_x, class_y=cls_y, freight_route=fr,
                                 transit_stop=tr, emergency_route=em, canopy_pct=canopy, env_burden_pctile=burden,
                                 collisions_5yr=int(coll), bike_gap=gap, data_source="user-entered (verify in field)"))
            st.success("Added as site CUSTOM.")
            sites = T.all_sites()
        except ValueError as e:
            st.error(str(e))

scan_w, goal_w = json.dumps(sw), json.dumps(gw)
scan = T.scan_intersections(scan_w, top_n=50)
options = {f"{c['site_id']} · {c['name']} (score {c['score']})": c["site_id"] for c in scan["candidates"]}
choice = st.selectbox("Intersection to redesign (ranked by opportunity score)", list(options))
site_id = options[choice]
site = T.get_site(site_id)

tab_scan, tab_design, tab_brief, tab_agent, tab_eval = st.tabs(
    ["① Scan", "② Redesign options", "③ Action brief", "④ Run the agent", "⑤ Evaluation"])

# ---------------------------------------------------------------- scan
with tab_scan:
    st.subheader("Where should we look first?")
    st.caption("Opportunity score = weighted sum of normalised indicators. High = worth investigating, "
               "not proven dangerous. Missing data is excluded and flagged, never guessed.")
    rows = []
    for c in scan["candidates"]:
        rows.append({"Site": c["site_id"], "Name": c["name"], "Score": c["score"], "Confidence": c["confidence"],
                     **{SCAN_LABELS[k]: v for k, v in c["components"].items()},
                     "Flags": ", ".join(c["flags"]), "Missing": ", ".join(c["missing_indicators"])})
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    contrib = pd.DataFrame({c["site_id"]: {SCAN_LABELS[k]: v for k, v in c["contributions"].items()}
                            for c in scan["candidates"]}).T
    st.bar_chart(contrib)
    geo = [{"lat": s.lat, "lon": s.lon} for s in sites if s.lat is not None and s.lon is not None]
    if geo:
        st.map(pd.DataFrame(geo))

# ---------------------------------------------------------------- design
ev = T.evaluate_redesign(site_id, goal_w)
with tab_design:
    st.subheader(f"Redesign options for {site.name}")
    st.image(ev["comparison_path"], width="stretch")
    if ev["ranked"]:
        best = ev["ranked"][0]
        st.success(f"Top option for your priorities: **{best['label']}** (score {best['score']}/100). "
                   "Move the priority sliders and watch the ranking change.")
        table = [{"Rank": r["rank"], "Design": r["label"], "Score": r["score"],
                  "Max ped. exposure (m)": r["metrics"]["max_ped_exposure_m"],
                  "Planting (m²)": r["metrics"]["green_m2"], "Protected bike (m)": r["metrics"]["protected_bike_m"],
                  "Pavement Δ (m²)": r["metrics"]["pavement_change_m2"],
                  "Calming (1-3)": r["metrics"]["traffic_calming_ordinal"],
                  "Rel. cost (1-5)": r["metrics"]["relative_cost_index"]} for r in ev["ranked"]]
        st.dataframe(pd.DataFrame(table), hide_index=True, width="stretch")
        st.caption(f"Existing: max pedestrian exposure {ev['baseline']['max_ped_exposure_m']} m, planting "
                   f"{ev['baseline']['green_m2']:.0f} m², protected bike {ev['baseline']['protected_bike_m']:.0f} m. "
                   "Metrics cover a ±40 m study window. Calming and cost are qualitative/relative placeholders.")
        st.markdown("**Why it ranked this way** (points contributed by each priority)")
        st.bar_chart(pd.DataFrame({r["label"]: {GOAL_LABELS[k]: v for k, v in r["contributions"].items()}
                                   for r in ev["ranked"]}).T)
        with st.expander("Warnings & assumptions for the top option"):
            for w in best["warnings"]:
                st.markdown(f"- ⚠️ {w}")
            for a in best["assumptions"]:
                st.markdown(f"- {a}")
    for i in ev["infeasible"]:
        st.error(f"**{i['label']}** screened out: " + " ".join(i["blockers"]))

# ---------------------------------------------------------------- brief
with tab_brief:
    b = T.generate_brief(site_id, goal_w)
    st.markdown(b["brief_markdown"])
    st.download_button("Download brief (.md)", b["brief_markdown"], file_name=f"{site_id}_brief.md")

# ---------------------------------------------------------------- agent
with tab_agent:
    st.subheader("Let the agent do it")
    ok, why = bedrock_available()
    mode = st.radio("Reasoning engine", ["Offline planner (no AWS needed)", "Amazon Bedrock via Strands Agents"],
                    help="Bedrock mode needs the strands-agents package, AWS credentials and Bedrock model access. "
                         "Set ECOEQUITY_MODEL_ID to choose a model enabled in your account.")
    if mode.startswith("Amazon") and not ok:
        st.info(f"Bedrock not ready: {why}. The offline planner will be used.")
    request = st.text_area("Request", "Find the intersection that most needs a redesign and recommend a layout.")
    if st.button("Run agent", type="primary"):
        with st.spinner("Agent working…"):
            try:
                out = run_agent(request, mode="bedrock" if mode.startswith("Amazon") and ok else "offline",
                                goals=gw, scan_weights=sw)
            except Exception as e:
                st.error(f"{type(e).__name__}: {e}")
                st.stop()
        if out.get("note"):
            st.info(out["note"])
        st.markdown(out["summary"])
        st.markdown("**Tool calls**")
        for i, t in enumerate(out["trace"], 1):
            st.markdown(f"{i}. `{t['tool']}({json.dumps(t['args'])})`  \n   ↳ {t['result_summary']}")
        arts = out.get("artifacts", {})
        if arts.get("sketch_png"):
            st.image(arts["sketch_png"], width="stretch")

# ---------------------------------------------------------------- evals
with tab_eval:
    st.subheader("Does the agent behave?")
    st.caption("Automated checks on geometry, constraints, ranking logic and honesty about data.")
    res = run_evals()
    st.metric("Checks passed", f"{sum(r['passed'] for r in res)}/{len(res)}")
    st.dataframe(pd.DataFrame([{"✓": "✅" if r["passed"] else "❌", "Check": r["check"], "Detail": r["detail"]}
                               for r in res]), hide_index=True, width="stretch")
