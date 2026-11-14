# EcoEquity AI: Reimagining Our Streets

An agentic tool that **scans intersections, compares redesigns, sketches them from measured geometry, and writes a community action brief**.
It is a concept tool for discussion, not an engineering design or traffic study.

## Quick start

```bash
pip install -r requirements.txt
streamlit run app.py          # interactive demo
python -m ecoequity.evals     # evaluation panel (19 checks)
```

The demo runs with **no AWS credentials** using a deterministic offline planner. See "Amazon Bedrock mode" below.

## How it works

One coordinating agent, four tools. **Python does all geometry and arithmetic; the LLM decides which tools to call and explains the results.**

| Tool | What it does |
|---|---|
| `scan_intersections` | Ranks candidate sites by an adjustable opportunity score (crossing width, bike gap, low canopy, environmental burden, collisions). Missing indicators are flagged, never guessed. |
| `propose_redesign` | Builds a geometry-based layout (shapely polygons) for one design and saves a before/after sketch. |
| `evaluate_redesign` | Compares all four designs, screens out infeasible ones with reasons, ranks the rest for the user's priorities. |
| `generate_brief` | One-page markdown brief with evidence, assumptions, review flags and next steps. |

Designs compared (it never defaults to a roundabout): **green roundabout**, **protected intersection**, **green curb extensions**, **neighborhood traffic circle**.

Reported metrics are measured from the polygons inside a ±40 m study window: planting area, protected bike length, pedestrian crossing exposure, pavement change.

## Using a real intersection

The sample sites in `data/sample_sites.csv` are **synthetic placeholders**. For a real one, use the sidebar form (or add a CSV row) with:
right-of-way and curb-to-curb widths for both streets, parking lane width, street class, and freight / transit / emergency flags.
Measure from aerial imagery and verify in the field. Optional scan indicators (canopy, collisions, environmental burden, bike gap) can come from
Oakland Open Data, OpenStreetMap and CalEnviroScreen.

## Amazon Bedrock mode (Strands Agents SDK)

Bedrock is strictly opt-in so ambient AWS credentials are never used by surprise:

```bash
export AWS_REGION=us-west-2
export ECOEQUITY_MODEL_ID=<a Bedrock model id enabled in your account>   # optional; otherwise the Strands default
export ECOEQUITY_USE_BEDROCK=1                                           # or pick Bedrock in the app's "Run the agent" tab
```

`ecoequity/agent.py` registers the four tools with a Strands `Agent` and a system prompt that forbids invented numbers, invented pollution
reductions, and defaulting to one design. **This path was not run end-to-end during development** (no Bedrock access in the build environment); the tool
schemas and agent construction were verified, the model call was not. Test it first.

## What it deliberately does not claim

- **No pollution-reduction percentages.** No validated exposure model is wired in; the brief describes pathways as uncertain.
- **Cost and traffic-calming are relative placeholders** (ordinal 1-5 / 1-3), not dollars or crash-reduction factors. Replace with City of Oakland / Caltrans
  unit costs and FHWA CMF Clearinghouse values.
- **Screening thresholds are heuristics** (e.g. roundabout inscribed-circle size, minimum lane widths). Verify against NACTO / FHWA / Caltrans guidance.
- Geometry is rectilinear (no curb radii, skewed legs, driveways, utilities). Turning paths, drainage, sight distance and ADA need a licensed engineer.
- Protected-intersection corner conflict detailing is not drawn or counted.

## Layout

```
app.py                      Streamlit UI
ecoequity/model.py          Site model + CSV loader
ecoequity/designs.py        Four redesign generators, feasibility rules, metrics
ecoequity/scoring.py        Scan score + weighted design ranking
ecoequity/render.py         Top-down sketches (matplotlib)
ecoequity/brief.py          Community action brief + image-model prompt
ecoequity/tools.py          The four agent tools
ecoequity/agent.py          Strands/Bedrock agent + offline planner
ecoequity/evals.py          Evaluation panel
data/sample_sites.csv       Synthetic placeholder sites
```

## Next steps

1. Enter your real intersection(s) and compare the sketch against aerial imagery.
2. Load real indicators (OSMnx street network, Oakland collision data) to replace the CSV; `Site` is the only schema to fill.
3. Add an image-model step that takes the brief's geometry-grounded prompt for a photorealistic before/after (the plan stays the source of truth).
4. Run Bedrock mode and add transcript-based evals (did the agent call the right tools, and quote only tool numbers?).
