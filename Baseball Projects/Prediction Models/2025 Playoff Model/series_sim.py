import argparse
import numpy as np
import pandas as pd
from typing import List, Dict
from statsmodels.miscmodels.ordinal_model import OrderedModel


def load_dataset(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    return df


def get_feature_columns(df: pd.DataFrame) -> List[str]:
    candidate_feats = [
        "pythW_pct",
        "run_diff_per_game",
        "wRC+",
        "xwOBA",
        "xFIP",
        "SIERA",
        "F_DRS",
        "F_OAA",
        "W",
        "L",
        "R",
        "RA",
    ]
    return [c for c in candidate_feats if c in df.columns]


def train_ordered_model(df: pd.DataFrame, feature_cols: List[str]):
    train = df[df["Year"] <= 2024].copy()
    X_tr = train[feature_cols].fillna(0.0)
    # Standardize to stabilize coefficients and margins
    means = X_tr.mean(axis=0)
    stds = X_tr.std(axis=0).replace(0.0, 1.0)
    X_tr_std = (X_tr - means) / stds
    y_tr = train["playoff_outcome"].astype(int)
    model = OrderedModel(y_tr, X_tr_std, distr="logit")
    res = model.fit(method="bfgs", maxiter=200, disp=False)
    return res, means, stds


def build_team_feature_map(
    df: pd.DataFrame,
    year: int,
    feature_cols: List[str],
    means: pd.Series,
    stds: pd.Series,
) -> Dict[str, np.ndarray]:
    year_df = df[df["Year"] == year].copy()
    year_df = year_df.dropna(subset=["Team"])  # safety
    team_to_vec: Dict[str, np.ndarray] = {}
    feats = year_df[feature_cols].fillna(0.0)
    feats = (feats - means) / stds.replace(0.0, 1.0)
    for team, row in zip(year_df["Team"].astype(str).values, feats.values):
        team_to_vec[team] = row.astype(float)
    return team_to_vec

def logistic(x: float) -> float:
    return 1.0 / (1.0 + np.exp(-x))

def head_to_head_game_win_prob(
    team_a: str,
    team_b: str,
    team_to_vec: Dict[str, np.ndarray],
    model_result,
    feature_cols: List[str],
    scale: float = 0.3,
    shrink: float = 0.6,
) -> float:
    if team_a not in team_to_vec:
        raise ValueError(f"Unknown team: {team_a}")
    if team_b not in team_to_vec:
        raise ValueError(f"Unknown team: {team_b}")
    beta = np.array([model_result.params.get(col, 0.0) for col in feature_cols], dtype=float)
    x_a = team_to_vec[team_a]
    x_b = team_to_vec[team_b]
    margin = float(np.dot(beta, (x_a - x_b)))
    p = float(logistic(scale * margin))
    p = 0.5 + shrink * (p - 0.5)
    return float(np.clip(p, 1e-3, 1 - 1e-3))


def simulate_best_of_series(p_game: float, best_of: int, n_sims: int = 10000, rng: np.random.Generator = None) -> float:
    if rng is None:
        rng = np.random.default_rng(42)
    wins_needed = best_of // 2 + 1
    wins_a = 0
    for _ in range(n_sims):
        a = 0
        b = 0
        while a < wins_needed and b < wins_needed:
            if rng.random() < p_game:
                a += 1
            else:
                b += 1
        if a >= wins_needed:
            wins_a += 1
    return wins_a / n_sims


def main():
    parser = argparse.ArgumentParser(description="Simulate MLB playoff series from matchup(s) using learned team strengths.")
    parser.add_argument("--csv", default="mlb_playoff_data_2016_2025.csv", help="Path to dataset CSV")
    parser.add_argument("--year", type=int, default=2025, help="Season year to use for team features")
    parser.add_argument("--teamA", type=str, help="Team A name as it appears in the CSV (column 'Team')")
    parser.add_argument("--teamB", type=str, help="Team B name as it appears in the CSV (column 'Team')")
    parser.add_argument("--best_of", type=int, default=5, choices=[3, 5, 7], help="Series length")
    parser.add_argument("--sims", type=int, default=20000, help="Monte Carlo simulations per series")
    parser.add_argument("--scale", type=float, default=0.3, help="Scale for mapping feature margin to game win prob")
    parser.add_argument("--shrink", type=float, default=0.6, help="Shrink per-game prob toward 0.5 (0..1)")
    parser.add_argument("--print_probs", action="store_true", help="Print single-game and series probabilities")

    args = parser.parse_args()

    df = load_dataset(args.csv)
    feature_cols = get_feature_columns(df)
    model_result, means, stds = train_ordered_model(df, feature_cols)

    if args.teamA and args.teamB:
        team_to_vec = build_team_feature_map(df, args.year, feature_cols, means, stds)
        p_game = head_to_head_game_win_prob(
            args.teamA,
            args.teamB,
            team_to_vec,
            model_result,
            feature_cols,
            scale=args.scale,
            shrink=args.shrink,
        )
        p_series = simulate_best_of_series(p_game, best_of=args.best_of, n_sims=args.sims)
        out = {
            "team_a": args.teamA,
            "team_b": args.teamB,
            "p_game_team_a": p_game,
            "p_series_team_a": p_series,
            "p_series_team_b": 1.0 - p_series,
        }
        if args.print_probs:
            print(
                f"Matchup: {out['team_a']} vs {out['team_b']} (best-of-{args.best_of})\n"
                f"  Per-game P({out['team_a']} win) = {out['p_game_team_a']:.3f}\n"
                f"  Series P({out['team_a']} win)   = {out['p_series_team_a']:.3f}\n"
                f"  Series P({out['team_b']} win)   = {out['p_series_team_b']:.3f}"
            )
        else:
            print(out)
    else:
        print("Provide --teamA and --teamB to simulate a series. Example:\n"
              "python series_sim.py --teamA Dodgers --teamB Braves --best_of 7 --print_probs")


if __name__ == "__main__":
    main()


