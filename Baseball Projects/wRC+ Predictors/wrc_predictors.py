"""
Rank every offensive stat by how well it predicts wRC+.

For each numeric offensive stat, fit a simple one-variable linear regression
against wRC+ across qualified hitter-seasons and report Pearson r (and r^2).
Prints the top 10 and saves a bar chart.

    python wrc_predictors.py                       # pulls 2021-2025 via pybaseball
    python wrc_predictors.py --start 2023 --end 2025
    python wrc_predictors.py --csv fangraphs.csv   # use a FanGraphs leaderboard export

If pybaseball can't reach FanGraphs (they sometimes block it with a 403), go to
the FanGraphs batting leaderboard, pick the seasons and columns you want,
click "Export Data", and pass that file with --csv.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

TARGET = "wRC+"

# Columns that are not offensive stats, or that are just wRC+ restated
# (wRC/wRAA/Batting runs/Off/WAR are built from the same wOBA-above-average
# number). Keeping them would make the ranking trivial.
EXCLUDE = {
    # identifiers / bookkeeping
    "IDfg", "playerid", "PlayerId", "xMLBAMID", "MLBAMID", "Season", "Age", "Name",
    "Team", "NameASCII", "Age Rng",
    # wRC+ restated
    "wRC+", "wRC", "wRAA", "Bat", "Batting", "Off", "RAR", "WAR", "Dollars",
    "WPA", "-WPA", "+WPA", "RE24", "REW", "pLI", "phLI", "PH", "WPA/LI", "Clutch",
    # defense / position
    "Def", "Fld", "Pos", "Rep", "Lg", "UBR", "wSB", "wGDP", "BsR", "Spd",
    # playing time, not skill
    "G", "AB", "PA", "Events",
}


def load_data(args) -> pd.DataFrame:
    if args.csv:
        return pd.read_csv(args.csv)
    from pybaseball import batting_stats
    return batting_stats(args.start, args.end, qual=args.min_pa)


def rank_predictors(df: pd.DataFrame, min_rows: int) -> pd.DataFrame:
    if TARGET not in df.columns:
        raise SystemExit(f"No '{TARGET}' column in the data.")
    # FanGraphs exports percentages like "12.3 %" as strings; make them numbers.
    for col in df.columns:
        if not pd.api.types.is_numeric_dtype(df[col]):
            cleaned = pd.to_numeric(df[col].astype(str).str.replace("%", "").str.strip(),
                                    errors="coerce")
            if cleaned.notna().mean() > 0.9:
                df[col] = cleaned

    y = df[TARGET].astype(float)
    rows = []
    for col in df.select_dtypes("number").columns:
        if col in EXCLUDE:
            continue
        x = df[col].astype(float)
        mask = x.notna() & y.notna()
        if mask.sum() < min_rows or x[mask].std() == 0:
            continue
        r = np.corrcoef(x[mask], y[mask])[0, 1]
        rows.append({"stat": col, "r": r, "r2": r * r, "n": int(mask.sum())})

    out = pd.DataFrame(rows)
    out["abs_r"] = out["r"].abs()
    return out.sort_values("abs_r", ascending=False).drop(columns="abs_r").reset_index(drop=True)


def plot(top: pd.DataFrame, path: Path, subtitle: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ink, ink2, surface, bar = "#0b0b0b", "#52514e", "#fcfcfb", "#2a78d6"
    top = top.iloc[::-1]  # best at the top
    fig, ax = plt.subplots(figsize=(8, 5.5), dpi=150, facecolor=surface)
    ax.set_facecolor(surface)
    ax.barh(top["stat"], top["r"].abs(), color=bar, height=0.6)
    for y_pos, (v, r) in enumerate(zip(top["r"].abs(), top["r"])):
        ax.text(v + 0.01, y_pos, f"r = {r:+.3f}", va="center", fontsize=9, color=ink2)
    ax.set_xlim(0, 1.12)
    ax.set_xlabel("|r| with wRC+", color=ink2)
    ax.tick_params(colors=ink2, length=0)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color("#d8d7d2")
    ax.grid(axis="x", color="#ecebe7", linewidth=0.8)
    ax.set_axisbelow(True)
    fig.suptitle("Top 10 stats for predicting wRC+", x=0.02, ha="left",
                 fontsize=14, fontweight="bold", color=ink)
    ax.set_title(subtitle, loc="left", fontsize=9, color=ink2)
    fig.tight_layout()
    fig.savefig(path, facecolor=surface)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--start", type=int, default=2021)
    ap.add_argument("--end", type=int, default=2025)
    ap.add_argument("--min-pa", type=int, default=300, help="PA minimum when pulling via pybaseball")
    ap.add_argument("--csv", help="FanGraphs batting leaderboard export to use instead of pybaseball")
    ap.add_argument("--top", type=int, default=10)
    ap.add_argument("--out", default=str(Path(__file__).parent / "wrc_predictors.png"))
    args = ap.parse_args()

    df = load_data(args)
    ranked = rank_predictors(df, min_rows=max(30, len(df) // 2))
    top = ranked.head(args.top)

    source = Path(args.csv).name if args.csv else f"{args.start}-{args.end}, {args.min_pa}+ PA"
    print(f"\nTop {args.top} predictors of wRC+  ({len(df)} hitter-seasons, {source})\n")
    print(f"{'Rank':<5}{'Stat':<14}{'r':>8}{'r^2':>8}")
    for i, row in top.iterrows():
        print(f"{i + 1:<5}{row['stat']:<14}{row['r']:>+8.3f}{row['r2']:>8.3f}")

    plot(top, Path(args.out), f"Pearson r across {len(df)} hitter-seasons ({source})")
    print(f"\nChart saved to {args.out}")


if __name__ == "__main__":
    main()
