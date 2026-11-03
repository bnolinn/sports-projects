# ---- Tweet-ready heatmaps ----
# Paste this into a NEW cell below your model code and run it after the
# model cells finish. It uses brew_roll and rays_roll from the model.
# To save a chart on your phone, long-press the image and tap Save.

import textwrap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

LEAGUE_XWOBA = 0.315   # white = about league average
SPREAD = 0.05          # colors max out at average +/- this much
MIN_COVERAGE = 0.6     # gray out cells where the model covered <60% of the pitcher's mix

SUBTITLE = ("Expected xwOBA vs pitches shaped like each pitcher's 2026 arsenal. "
            "Red = hitter edge, blue = pitcher edge.")
FOOTER = ("Data: Baseball Savant via pybaseball. Hitter data 2021+, pitcher profiles 2026, "
          "through Oct 19. Gray = not enough data.")


def tweet_heatmap(roll, title, filename):
    if roll is None or roll.empty:
        print(f"No data for {title}")
        return
    d = roll.copy()
    d.loc[d["coverage"] < MIN_COVERAGE, "xwOBA_shrunk"] = np.nan
    piv = d.pivot(index="batter", columns="pitcher", values="xwOBA_shrunk")
    piv = piv.dropna(how="all").dropna(axis=1, how="all")

    # Best hitters on top, toughest pitchers on the left
    piv = piv.loc[piv.mean(axis=1).sort_values(ascending=False).index,
                  piv.mean(axis=0).sort_values().index]
    vals = piv.to_numpy(float)
    n_rows, n_cols = vals.shape

    norm = TwoSlopeNorm(vcenter=LEAGUE_XWOBA,
                        vmin=LEAGUE_XWOBA - SPREAD, vmax=LEAGUE_XWOBA + SPREAD)
    cmap = plt.get_cmap("RdBu_r").copy()
    cmap.set_bad("#e9e9e9")

    fig, ax = plt.subplots(figsize=(0.95 * n_cols + 3.2, 0.55 * n_rows + 2.6), dpi=200)
    im = ax.imshow(np.ma.masked_invalid(vals), cmap=cmap, norm=norm, aspect="auto")

    # Cell labels: .342 style, white text on the dark cells so it stays readable
    for i in range(n_rows):
        for j in range(n_cols):
            v = vals[i, j]
            if np.isnan(v):
                continue
            dark = abs(v - LEAGUE_XWOBA) > SPREAD * 0.6
            ax.text(j, i, f"{v:.3f}".lstrip("0"), ha="center", va="center",
                    fontsize=8, color="white" if dark else "#222")

    ax.set_xticks(range(n_cols))
    ax.set_xticklabels(piv.columns, rotation=45, ha="right", fontsize=9)
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels(piv.index, fontsize=9)
    ax.set_xticks(np.arange(-0.5, n_cols, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n_rows, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.5)
    ax.tick_params(which="both", length=0)
    for s in ax.spines.values():
        s.set_visible(False)

    cb = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cb.set_label("Expected xwOBA", fontsize=8)
    cb.ax.tick_params(labelsize=7)
    cb.outline.set_visible(False)

    width_chars = int(fig.get_figwidth() * 13)
    fig.suptitle(title, x=0.02, ha="left", fontsize=14, fontweight="bold")
    fig.text(0.02, 0.945, textwrap.fill(SUBTITLE, width_chars), ha="left", va="top",
             fontsize=8.5, color="#555")
    fig.text(0.02, 0.01, textwrap.fill(FOOTER, width_chars + 10), ha="left", va="bottom",
             fontsize=7, color="#888")
    fig.tight_layout(rect=(0, 0.04, 1, 0.92))
    fig.savefig(filename, dpi=200, bbox_inches="tight", facecolor="white")
    plt.show()


tweet_heatmap(brew_roll, "Brewers hitters vs Rays pitchers", "brewers_vs_rays.png")
tweet_heatmap(rays_roll, "Rays hitters vs Brewers pitchers", "rays_vs_brewers.png")