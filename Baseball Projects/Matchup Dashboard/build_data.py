"""
Download a season of pitch-level Statcast data and save it as a compact parquet
file that the dashboard reads.

    python build_data.py --year 2025

Data is pulled in ~weekly chunks and each chunk is cached under data/chunks/,
so an interrupted run resumes where it stopped.
"""
import argparse
import datetime as dt
from pathlib import Path

import pandas as pd
from pybaseball import statcast, playerid_reverse_lookup

DATA_DIR = Path(__file__).parent / "data"

KEEP_COLS = [
    "game_date", "game_type", "game_pk", "at_bat_number", "pitch_number",
    "batter", "pitcher", "stand", "p_throws", "home_team", "away_team", "inning_topbot",
    "pitch_type", "pitch_name", "release_speed", "release_spin_rate", "spin_axis",
    "pfx_x", "pfx_z", "release_pos_x", "release_pos_z", "release_extension", "arm_angle",
    "plate_x", "plate_z", "sz_top", "sz_bot", "zone", "balls", "strikes",
    "description", "events", "type", "bb_type", "launch_speed", "launch_angle",
    "estimated_woba_using_speedangle", "woba_value", "woba_denom", "delta_run_exp",
    "bat_speed", "swing_length",
]

# Rough season windows (spring training excluded, postseason included).
SEASON_WINDOWS = {
    2023: ("2023-03-30", "2023-11-02"),
    2024: ("2024-03-20", "2024-10-31"),
    2025: ("2025-03-18", "2025-11-02"),
    2026: ("2026-03-25", "2026-11-05"),
}


def chunks(start: str, end: str, days: int = 7):
    s = dt.date.fromisoformat(start)
    e = dt.date.fromisoformat(end)
    while s <= e:
        c_end = min(s + dt.timedelta(days=days - 1), e)
        yield s.isoformat(), c_end.isoformat()
        s = c_end + dt.timedelta(days=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", type=int, default=2025)
    ap.add_argument("--start", help="override start date YYYY-MM-DD")
    ap.add_argument("--end", help="override end date YYYY-MM-DD")
    args = ap.parse_args()

    start, end = SEASON_WINDOWS.get(args.year, (f"{args.year}-03-20", f"{args.year}-11-05"))
    start = args.start or start
    end = args.end or end

    chunk_dir = DATA_DIR / "chunks" / str(args.year)
    chunk_dir.mkdir(parents=True, exist_ok=True)

    for c_start, c_end in chunks(start, end):
        out = chunk_dir / f"{c_start}_{c_end}.parquet"
        if out.exists():
            continue
        print(f"Fetching {c_start} -> {c_end} ...", flush=True)
        df = statcast(start_dt=c_start, end_dt=c_end, verbose=False, parallel=True)
        if df is None or df.empty:
            pd.DataFrame(columns=KEEP_COLS).to_parquet(out)
            continue
        cols = [c for c in KEEP_COLS if c in df.columns]
        df[cols].to_parquet(out, index=False)
        print(f"  {len(df):,} pitches", flush=True)

    parts = [pd.read_parquet(p) for p in sorted(chunk_dir.glob("*.parquet"))]
    df = pd.concat([p for p in parts if not p.empty], ignore_index=True)
    df = df[df["game_type"].isin(["R", "F", "D", "L", "W"])]
    df = df.drop_duplicates(subset=["game_pk", "at_bat_number", "pitch_number"])

    # Player names from the Chadwick register (statcast's player_name is ambiguous).
    ids = pd.unique(pd.concat([df["batter"], df["pitcher"]]).dropna().astype(int))
    lk = playerid_reverse_lookup(list(ids), key_type="mlbam")
    lk["name"] = lk["name_first"].str.title() + " " + lk["name_last"].str.title()
    names = lk[["key_mlbam", "name"]].rename(columns={"key_mlbam": "player_id"})

    DATA_DIR.mkdir(exist_ok=True)
    df.to_parquet(DATA_DIR / f"statcast_{args.year}.parquet", index=False)
    names.to_parquet(DATA_DIR / f"players_{args.year}.parquet", index=False)
    print(f"Saved {len(df):,} pitches and {len(names):,} players for {args.year}.")


if __name__ == "__main__":
    main()
