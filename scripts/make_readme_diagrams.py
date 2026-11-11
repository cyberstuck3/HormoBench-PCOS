"""
Render the two README diagrams as PNGs.

GitHub's Mermaid renderer clips node labels in some browsers, so the README
embeds these static images instead.

    python scripts/make_readme_diagrams.py   # → docs/figures/*.png
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUT = Path(__file__).resolve().parents[1] / "docs" / "figures"

INK = "#1f2937"
MUTED = "#6b7280"
EDGE = "#94a3b8"
GROUPS = {  # fill, border
    "prep":   ("#eff6ff", "#3b82f6"),
    "model":  ("#f5f3ff", "#7c3aed"),
    "report": ("#ecfdf5", "#059669"),
    "test":   ("#fff7ed", "#ea580c"),
}


def box(ax, x, y, w, h, title, sub, group):
    fill, border = GROUPS[group]
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.12",
                                fc=fill, ec=border, lw=1.6))
    if sub:
        ax.text(x + w / 2, y + h * 0.63, title, ha="center", va="center",
                fontsize=12, fontweight="bold", color=INK)
        ax.text(x + w / 2, y + h * 0.30, sub, ha="center", va="center",
                fontsize=9.5, color=MUTED)
    else:
        ax.text(x + w / 2, y + h / 2, title, ha="center", va="center",
                fontsize=11, color=INK)


def arrow(ax, a, b, label=None, rad=0.0):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=14,
                                 color=EDGE, lw=1.6, connectionstyle=f"arc3,rad={rad}"))
    if label:
        ax.text((a[0] + b[0]) / 2, (a[1] + b[1]) / 2 + 0.22, label, ha="center",
                va="bottom", fontsize=9.5, color=MUTED, style="italic")


def canvas(w, h):
    fig, ax = plt.subplots(figsize=(w, h), dpi=200)
    ax.set_xlim(0, w)
    ax.set_ylim(0, h)
    ax.axis("off")
    fig.patch.set_facecolor("white")
    return fig, ax


def pipeline():
    rows = [
        ("prep", "1  PREPARE", [("Raw data", "Kerala xlsx"), ("Clean", "deterministic"),
                                ("Split", "70/15/15, seed 42")]),
        ("model", "2  MODEL", [("Train", "4 models, 5-fold CV"), ("Validate", "threshold 0.21"),
                               ("Test", "single pass")]),
        ("report", "3  REPORT", [("Rigor", "bootstrap, DeLong, Platt"), ("Explain", "SHAP + literature"),
                                 ("Publish", "tables, figures")]),
    ]
    W, H = 10.5, 5.4
    fig, ax = canvas(W, H)
    bw, bh, gap, x0 = 2.45, 1.0, 0.55, 1.75
    for r, (group, label, items) in enumerate(rows):
        y = H - 0.35 - bh - r * (bh + 0.7)
        ax.text(0.2, y + bh / 2, label, ha="left", va="center", fontsize=10.5,
                fontweight="bold", color=GROUPS[group][1])
        for i, (t, s) in enumerate(items):
            x = x0 + i * (bw + gap)
            box(ax, x, y, bw, bh, t, s, group)
            if i:
                arrow(ax, (x - gap + 0.05, y + bh / 2), (x - 0.05, y + bh / 2))
        if r < len(rows) - 1:  # last box of this row → first box of next row
            x_last = x0 + 2 * (bw + gap) + bw / 2
            arrow(ax, (x_last, y - 0.05), (x0 + bw / 2, y - 0.65), rad=-0.08)
    fig.savefig(OUT / "pipeline.png", bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)


def anti_leakage():
    cols = [
        ("prep", "TRAIN  70%", ["fit preprocessor", "fit models via CV"]),
        ("model", "VALIDATION  15%", ["choose winner", "tune threshold 0.21", "fit Platt scaling"]),
        ("test", "TEST  15%  (sealed)", ["single-pass evaluation", "bootstrap 95% CI", "DeLong pairwise"]),
    ]
    W, H = 10.4, 3.4
    fig, ax = canvas(W, H)
    cw, gap, x0, top = 2.7, 1.0, 0.1, H - 0.1
    ch = H - 0.2
    for c, (group, title, items) in enumerate(cols):
        fill, border = GROUPS[group]
        x = x0 + c * (cw + gap)
        ax.add_patch(FancyBboxPatch((x, top - ch), cw, ch, boxstyle="round,pad=0,rounding_size=0.15",
                                    fc=fill, ec=border, lw=1.8))
        ax.text(x + cw / 2, top - 0.38, title, ha="center", va="center",
                fontsize=12, fontweight="bold", color=border)
        for i, item in enumerate(items):
            iy = top - 1.25 - i * 0.68
            ax.add_patch(FancyBboxPatch((x + 0.2, iy - 0.25), cw - 0.4, 0.5,
                                        boxstyle="round,pad=0,rounding_size=0.08",
                                        fc="white", ec=border, lw=1))
            ax.text(x + cw / 2, iy, item, ha="center", va="center", fontsize=10.5, color=INK)
        if c:
            mid = top - ch / 2
            arrow(ax, (x - gap + 0.08, mid), (x - 0.08, mid))
            ax.text(x - gap / 2, mid + 0.18, "transform\nonly", ha="center", va="bottom",
                    fontsize=8.5, color=MUTED, style="italic")
    fig.savefig(OUT / "anti_leakage.png", bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    pipeline()
    anti_leakage()
    print(f"✓ Diagrams → {OUT}")


if __name__ == "__main__":
    main()
