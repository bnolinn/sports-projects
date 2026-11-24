"""
Rank every offensive stat by how well it predicts wOBA.

wOBA is the stat wRC+ is built on (wRC+ is wOBA turned into runs, then park-
and league-adjusted), and unlike wRC+ it is published by Baseball Savant, which
offers a free CSV leaderboard. For each numeric offensive stat this script fits a
one-variable linear regression against wOBA across qualified hitter-seasons and
reports Pearson r (and r^2). It prints the top 10 and saves a bar chart.

    python wrc_predictors.py                        # 2021-2025, qualified hitters
    python wrc_predictors.py --start 2023 --end 2025
    python wrc_predictors.py --csv savant.csv       # use a saved Savant export

Downloaded data is cached in data/ next to this script.
"""
import argparse
import io
from pathlib import Path

import numpy as np
import pandas as pd
import requests

HERE = Path(__file__).parent
TARGET = "woba"

SAVANT_URL = "https://baseballsavant.mlb.com/leaderboard/custom"

# Savant column -> readable label. These are every offensive stat requested.
STATS = {
    "hit": "H", "single": "1B", "double": "2B", "triple": "3B", "home_run": "HR",
    "strikeout": "SO", "walk": "BB", "b_hit_by_pitch": "HBP", "b_rbi": "RBI", "r_run": "R",
    "r_total_stolen_base": "SB",
    "k_percent": "K%", "bb_percent": "BB%", "batting_avg": "AVG", "on_base_percent": "OBP",
    "slg_percent": "SLG", "on_base_plus_slg": "OPS", "isolated_power": "ISO", "babip": "BABIP",
    "xba": "xBA", "xslg": "xSLG", "xwoba": "xwOBA", "xobp": "xOBP", "xiso": "xISO",
    "wobacon": "wOBAcon", "xwobacon": "xwOBAcon", "bacon": "BACON", "xbacon": "xBACON",
    "exit_velocity_avg": "Avg EV", "launch_angle_avg": "Avg LA", "sweet_spot_percent": "Sweet Spot%",
    "barrel_batted_rate": "Barrel%", "barrel": "Barrels", "solidcontact_percent": "Solid Contact%",
    "flareburner_percent": "Flare/Burner%", "poorlyunder_percent": "Under%",
    "poorlytopped_percent": "Topped%", "poorlyweak_percent": "Weak%", "hard_hit_percent": "Hard Hit%",
    "avg_best_speed": "Best Speed", "avg_hyper_speed": "Adj. EV",
    "z_swing_percent": "Z-Swing%", "z_swing_miss_percent": "Z-Whiff%", "oz_swing_percent": "Chase%",
    "oz_swing_miss_percent": "O-Whiff%", "oz_contact_percent": "O-Contact%",
    "iz_contact_percent": "Z-Contact%", "whiff_percent": "Whiff%", "swing_percent": "Swing%",
    "meatball_swing_percent": "Meatball Swing%", "pull_percent": "Pull%",
    "straightaway_percent": "Straight%", "opposite_percent": "Oppo%", "groundballs_percent": "GB%",
    "flyballs_percent": "FB%", "linedrives_percent": "LD%", "popups_percent": "PU%",
    "avg_swing_speed": "Bat Speed", "fast_swing_rate": "Fast Swing%", "blasts_contact": "Blast/Contact",
    "blasts_swing": "Blast/Swing", "squared_up_contact": "Squared Up/Contact",
    "squared_up_swing": "Squared Up/Swing", "avg_swing_length": "Swing Length",
    "attack_angle": "Attack Angle", "ideal_angle_rate": "Ideal Attack Angle%",
    "sprint_speed": "Sprint Speed",
}


def fetch_savant(start: int, end: int, min_pa: str) -> pd.DataFrame:
    cache = HERE / "data" / f"savant_batters_{start}_{end}_min{min_pa}.csv"
    if cache.exists():
        return pd.read_csv(cache)
    params = {
        "year": ",".join(str(y) for y in range(start, end + 1)),
        "type": "batter", "filter": "", "min": min_pa,
        "selections": ",".join(["pa", TARGET, *STATS]),
        "chart": "false", "csv": "true",
    }
    resp = requests.get(SAVANT_URL, params=params, timeout=60)
    resp.raise_for_status()
    df = pd.read_csv(io.StringIO(resp.content.decode("utf-8-sig")))
    cache.parent.mkdir(exist_ok=True)
    df.to_csv(cache, index=False)
    return df


def rank_predictors(df: pd.DataFrame, min_rows: int) -> pd.DataFrame:
    if TARGET not in df.columns:
        raise SystemExit(f"No '{TARGET}' column in the data.")
    y = pd.to_numeric(df[TARGET], errors="coerce")
    rows = []
    for col, label in STATS.items():
        if col not in df.columns:
            continue
        x = pd.to_numeric(df[col], errors="coerce")
        mask = x.notna() & y.notna()
        if mask.sum() < min_rows or x[mask].std() == 0:
            continue
        r = np.corrcoef(x[mask], y[mask])[0, 1]
        rows.append({"stat": label, "r": r, "r2": r * r, "n": int(mask.sum())})
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
    ax.set_xlabel("|r| with wOBA", color=ink2)
    ax.tick_params(colors=ink2, length=0)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color("#d8d7d2")
    ax.grid(axis="x", color="#ecebe7", linewidth=0.8)
    ax.set_axisbelow(True)
    fig.suptitle("Top 10 stats for predicting wOBA", x=0.02, ha="left",
                 fontsize=14, fontweight="bold", color=ink)
    ax.set_title(subtitle, loc="left", fontsize=9, color=ink2)
    fig.tight_layout()
    fig.savefig(path, facecolor=surface)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--start", type=int, default=2021)
    ap.add_argument("--end", type=int, default=2025)
    ap.add_argument("--min-pa", default="q", help="'q' for qualified hitters, or a PA number")
    ap.add_argument("--csv", help="Baseball Savant custom-leaderboard export to use instead of downloading")
    ap.add_argument("--top", type=int, default=10)
    ap.add_argument("--out", default=str(HERE / "wrc_predictors.png"))
    args = ap.parse_args()

    df = pd.read_csv(args.csv) if args.csv else fetch_savant(args.start, args.end, args.min_pa)
    ranked = rank_predictors(df, min_rows=max(30, len(df) // 2))
    top = ranked.head(args.top)

    pa = "qualified" if args.min_pa == "q" else f"{args.min_pa}+ PA"
    source = Path(args.csv).name if args.csv else f"{args.start}-{args.end}, {pa}, Baseball Savant"
    print(f"\nTop {args.top} predictors of wOBA  ({len(df)} hitter-seasons, {source})\n")
    print(f"{'Rank':<5}{'Stat':<20}{'r':>8}{'r^2':>8}{'n':>6}")
    for i, row in top.iterrows():
        print(f"{i + 1:<5}{row['stat']:<20}{row['r']:>+8.3f}{row['r2']:>8.3f}{row['n']:>6}")

    plot(top, Path(args.out), f"Pearson r across {len(df)} hitter-seasons ({source})")
    print(f"\nChart saved to {args.out}")


if __name__ == "__main__":
    main()
