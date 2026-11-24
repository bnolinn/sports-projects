"""
Rank every offensive stat by how well it predicts a hitter's NEXT-season wOBA.

Same-season correlations mostly reward stats built from the same hits and walks
as wOBA (OPS, SLG, OBP), which is circular. Asking "which stat this year best
predicts wOBA next year" rewards stats that measure repeatable skill and punishes
ones driven by luck, which is what matters when deciding which stats to trust.

wOBA is used as the target because it is the stat wRC+ is built on and, unlike
wRC+, Baseball Savant publishes it in a free CSV leaderboard. For each stat this
script pairs a hitter's year-N value with his year-N+1 wOBA (qualified both
seasons), fits a one-variable linear regression, and reports Pearson r (and r^2).
wOBA itself is included as a baseline. It prints the top 10 and saves a bar chart.

    python wrc_predictors.py                        # 2015-2025 (Statcast era)
    python wrc_predictors.py --start 2019 --end 2025
    python wrc_predictors.py --include-2020         # keep the 60-game season
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
    "woba": "wOBA (baseline)",
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
        "selections": ",".join(["pa", *STATS]),
        "chart": "false", "csv": "true",
    }
    resp = requests.get(SAVANT_URL, params=params, timeout=60)
    resp.raise_for_status()
    df = pd.read_csv(io.StringIO(resp.content.decode("utf-8-sig")))
    cache.parent.mkdir(exist_ok=True)
    df.to_csv(cache, index=False)
    return df


def next_season_pairs(df: pd.DataFrame, include_2020: bool) -> pd.DataFrame:
    """One row per hitter per season N, with his season N+1 wOBA attached."""
    if not include_2020:
        df = df[df["year"] != 2020]  # 60-game season: tiny samples, and 2019 -> 2021 isn't "next season"
    nxt = df[["player_id", "year", TARGET]].rename(columns={TARGET: "next_woba"})
    nxt["year"] -= 1
    return df.merge(nxt, on=["player_id", "year"])


def rank_predictors(pairs: pd.DataFrame, min_rows: int) -> tuple[pd.DataFrame, list[str]]:
    y = pd.to_numeric(pairs["next_woba"], errors="coerce")
    rows, skipped = [], []
    for col, label in STATS.items():
        if col not in pairs.columns:
            continue
        x = pd.to_numeric(pairs[col], errors="coerce")
        mask = x.notna() & y.notna()
        # Stats that only exist for recent seasons (bat tracking) are dropped, so every
        # stat is scored on the same hitters; r from a different sample isn't comparable.
        if mask.sum() < min_rows or x[mask].std() == 0:
            skipped.append(label)
            continue
        r = np.corrcoef(x[mask], y[mask])[0, 1]
        rows.append({"stat": label, "r": r, "r2": r * r, "n": int(mask.sum())})
    out = pd.DataFrame(rows)
    out["abs_r"] = out["r"].abs()
    out = out.sort_values("abs_r", ascending=False).drop(columns="abs_r").reset_index(drop=True)
    return out, skipped


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
    ax.set_xlim(0, min(1.0, top["r"].abs().max() * 1.25))
    ax.set_xlabel("|r| with next season's wOBA", color=ink2)
    ax.tick_params(colors=ink2, length=0)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color("#d8d7d2")
    ax.grid(axis="x", color="#ecebe7", linewidth=0.8)
    ax.set_axisbelow(True)
    fig.suptitle("Which hitting stats best predict next season's wOBA?", x=0.02, ha="left",
                 fontsize=14, fontweight="bold", color=ink)
    ax.set_title(subtitle, loc="left", fontsize=9, color=ink2)
    fig.tight_layout()
    fig.savefig(path, facecolor=surface)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--start", type=int, default=2015)
    ap.add_argument("--end", type=int, default=2025)
    ap.add_argument("--min-pa", default="q", help="'q' for qualified hitters, or a PA number")
    ap.add_argument("--include-2020", action="store_true", help="keep the shortened 2020 season")
    ap.add_argument("--csv", help="Baseball Savant custom-leaderboard export to use instead of downloading")
    ap.add_argument("--top", type=int, default=10)
    ap.add_argument("--out", default=str(HERE / "wrc_predictors.png"))
    args = ap.parse_args()

    df = pd.read_csv(args.csv) if args.csv else fetch_savant(args.start, args.end, args.min_pa)
    pairs = next_season_pairs(df, args.include_2020)
    ranked, skipped = rank_predictors(pairs, min_rows=int(0.9 * len(pairs)))
    top = ranked.head(args.top)

    pa = "qualified both seasons" if args.min_pa == "q" else f"{args.min_pa}+ PA both seasons"
    years = f"{df['year'].min()}-{df['year'].max()}"
    source = Path(args.csv).name if args.csv else f"{years}, {pa}, Baseball Savant"
    print(f"\nTop {args.top} predictors of next-season wOBA  ({len(pairs)} season pairs, {source})\n")
    print(f"{'Rank':<5}{'Stat':<20}{'r':>8}{'r^2':>8}{'n':>6}")
    for i, row in top.iterrows():
        print(f"{i + 1:<5}{row['stat']:<20}{row['r']:>+8.3f}{row['r2']:>8.3f}{row['n']:>6}")

    for label in ("AVG", "OBP", "BABIP"):  # common stats worth seeing even if they miss the top 10
        hit = ranked[ranked["stat"] == label]
        if not hit.empty and label not in set(top["stat"]):
            row = hit.iloc[0]
            print(f"{hit.index[0] + 1:<5}{label:<20}{row['r']:>+8.3f}{row['r2']:>8.3f}{row['n']:>6}")

    if skipped:
        print(f"\nNot ranked (missing for some seasons in range): {', '.join(skipped)}")

    plot(top, Path(args.out), f"Pearson r: this season's stat vs. next season's wOBA\n{len(pairs)} season pairs, {source}")
    print(f"\nChart saved to {args.out}")


if __name__ == "__main__":
    main()
