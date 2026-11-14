"""Top-down concept sketches rendered from the same polygons the metrics use."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import PathPatch
from matplotlib.path import Path as MPath
from shapely.geometry import MultiPolygon, Polygon

from .designs import Layout, metrics
from .model import Site

COLORS = {
    "bg": "#f4f1ea", "sidewalk": "#e6e0d2", "pavement": "#6b7078", "green": "#4f9d5d",
    "bike": "#2a9d8f", "hard": "#b8b2a2", "cross": "#ffffff", "row": "#9a8f78",
}


def _polys(geom):
    if geom is None or geom.is_empty:
        return []
    if isinstance(geom, Polygon):
        return [geom]
    if isinstance(geom, MultiPolygon):
        return list(geom.geoms)
    return [g for g in getattr(geom, "geoms", []) if isinstance(g, (Polygon, MultiPolygon))
            for g in _polys(g)]


def _patch(poly: Polygon, **kw):
    verts, codes = [], []
    for ring in [poly.exterior, *poly.interiors]:
        xy = list(ring.coords)
        verts += xy
        codes += [MPath.MOVETO] + [MPath.LINETO] * (len(xy) - 2) + [MPath.CLOSEPOLY]
    return PathPatch(MPath(verts, codes), **kw)


def _draw(ax, geom, **kw):
    for p in _polys(geom):
        ax.add_patch(_patch(p, **kw))


def draw_layout(ax, site: Site, lay: Layout, extent: float = 30.0, title: str | None = None, base: Layout | None = None):
    ax.set_facecolor(COLORS["bg"])
    _draw(ax, lay.row, facecolor=COLORS["sidewalk"], edgecolor=COLORS["row"], linewidth=0.8, linestyle="--", zorder=1)
    _draw(ax, lay.pavement, facecolor=COLORS["pavement"], edgecolor="none", zorder=2)
    _draw(ax, lay.hardscape, facecolor=COLORS["hard"], edgecolor="#8a8474", linewidth=0.5, zorder=3)
    _draw(ax, lay.bike, facecolor=COLORS["bike"], edgecolor="none", alpha=0.95, zorder=4)
    _draw(ax, lay.green, facecolor=COLORS["green"], edgecolor="#2f6b3b", linewidth=0.5, zorder=5)
    for poly, exp in lay.crosswalks:
        # zebra stripes: draw the crosswalk polygon as a hatch
        _draw(ax, poly.intersection(lay.row), facecolor="none", edgecolor=COLORS["cross"], hatch="////",
              linewidth=0.0, zorder=6)
    ax.set_xlim(-extent, extent)
    ax.set_ylim(-extent, extent)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_color("#cfc8b8")
    t = title or lay.label
    ax.set_title(t, fontsize=11, fontweight="bold", loc="left")
    # scale bar
    ax.plot([-extent + 2, -extent + 12], [-extent + 2.5, -extent + 2.5], color="#222", lw=2, zorder=9)
    ax.text(-extent + 7, -extent + 3.5, "10 m", ha="center", fontsize=7, zorder=9)
    ax.annotate("N", xy=(extent - 3, extent - 3), xytext=(extent - 3, extent - 8), ha="center", fontsize=8,
                arrowprops=dict(arrowstyle="->", color="#222"), zorder=9)
    if lay.feasible and base is not None:
        m = metrics(lay, base)
        txt = (f"green {m['green_m2']:.0f} m²  ·  protected bike {m['protected_bike_m']:.0f} m\n"
               f"max ped. exposure {m['max_ped_exposure_m']:.1f} m  ·  pavement {m['pavement_change_m2']:+.0f} m²")
        ax.set_xlabel(txt, fontsize=7.5, color="#333")
    elif not lay.feasible:
        import textwrap
        reason = "\n".join(textwrap.fill(b, 44) for b in lay.blockers[:1])
        ax.text(0, 0, "NOT FEASIBLE HERE\n" + reason, ha="center", va="center",
                fontsize=8, color="#7a1f1f", bbox=dict(fc="white", ec="#7a1f1f", alpha=0.92), zorder=10)


def legend(fig):
    from matplotlib.patches import Patch
    items = [Patch(fc=COLORS["pavement"], label="Roadway"), Patch(fc=COLORS["sidewalk"], ec=COLORS["row"], label="Sidewalk / ROW"),
             Patch(fc=COLORS["green"], label="Planting"), Patch(fc=COLORS["bike"], label="Protected bike path"),
             Patch(fc=COLORS["hard"], label="Islands / separators"), Patch(fc="white", ec="#999", hatch="////", label="Crosswalk")]
    fig.legend(handles=items, loc="outside lower center", ncol=6, fontsize=8, frameon=False)


def render_single(site: Site, lay: Layout, base: Layout, path: str | Path) -> str:
    fig, axes = plt.subplots(1, 2, figsize=(11, 6.2), layout="constrained")
    draw_layout(axes[0], site, base, title="Before: existing layout", base=base)
    draw_layout(axes[1], site, lay, title=f"After: {lay.label}", base=base)
    fig.suptitle(f"{site.name} - CONCEPT SKETCH, not for construction", fontsize=10, color="#7a1f1f")
    legend(fig)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return str(path)


def render_comparison(site: Site, layouts: dict[str, Layout], path: str | Path) -> str:
    keys = ["existing"] + [k for k in layouts if k != "existing"]
    cols = 3
    rows = (len(keys) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5.2 * cols, 5.9 * rows), layout="constrained")
    axes = axes.flatten() if hasattr(axes, "flatten") else [axes]
    base = layouts["existing"]
    for ax, k in zip(axes, keys):
        draw_layout(ax, site, layouts[k], base=base, title=layouts[k].label)
    for ax in axes[len(keys):]:
        ax.axis("off")
    fig.suptitle(f"{site.name} - CONCEPT SKETCHES, not for construction", fontsize=11, color="#7a1f1f")
    legend(fig)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return str(path)
