"""
Pitch-level matchup engine.

Core idea:
Every pitch is described by its physical characteristics (velocity, induced
vertical break, arm-side horizontal break, spin, release height/side, extension).
For a batter-vs-pitcher matchup we, for each pitch in the pitcher's arsenal:

1. League "stuff" baseline: kernel-weight every league pitch (same pitcher hand,
   same batter side) by how physically similar it is to this pitch, and take the
   weighted league outcome. That is what an average hitter does vs this shape.
2. Pitcher estimate: the pitcher's actual results with this pitch vs this batter
   side, shrunk toward the stuff baseline (Bayesian, pseudo-counts).
3. Batter effect: the same kernel applied to the pitches this batter has seen
   (from same-handed pitchers). His weighted results, shrunk toward the
   baseline, minus the baseline = how much better/worse than average he is vs
   pitches *shaped like this one*, regardless of what they were labeled.
4. Combined expectation = pitcher estimate + batter effect (log-odds for rates).

The matchup is the usage-weighted sum across the arsenal. Run value is
count-neutralized (each pitch's delta run expectancy minus the league average
for that ball-strike count), expressed per 100 pitches from the batter's view:
positive = batter advantage, negative = pitcher advantage.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).parent / "data"

PITCH_NAMES = {
    "FF": "4-Seam", "SI": "Sinker", "FC": "Cutter", "SL": "Slider", "ST": "Sweeper",
    "SV": "Slurve", "CU": "Curveball", "KC": "Knuckle Curve", "CS": "Slow Curve",
    "CH": "Changeup", "FS": "Splitter", "FO": "Forkball", "SC": "Screwball",
    "KN": "Knuckleball", "EP": "Eephus", "FA": "Fastball",
}
PITCH_GROUP = {
    "FF": "Fastball", "SI": "Fastball", "FC": "Fastball", "FA": "Fastball",
    "SL": "Breaking", "ST": "Breaking", "SV": "Breaking", "CU": "Breaking",
    "KC": "Breaking", "CS": "Breaking",
    "CH": "Offspeed", "FS": "Offspeed", "FO": "Offspeed", "SC": "Offspeed",
    "KN": "Offspeed", "EP": "Offspeed",
}
EXCLUDE_TYPES = {"PO", "IN", "AB", "AS", "UN", "FA"}

SWING_DESC = {
    "swinging_strike", "swinging_strike_blocked", "foul", "foul_tip", "hit_into_play",
    "foul_bunt", "missed_bunt", "bunt_foul_tip",
}
WHIFF_DESC = {"swinging_strike", "swinging_strike_blocked", "foul_tip", "missed_bunt"}
CALLED_STRIKE = {"called_strike"}

# Similarity kernel: feature > steps
KERNEL_FEATURES = {
    "release_speed": 3.0,    # mph
    "ivb": 4.5,              # inches induced vertical break
    "hb_arm": 4.5,           # inches arm-side horizontal break
    "release_spin_rate": 500.0,  # rpm
    "release_pos_z": 1.0,    # ft
    "rel_x_arm": 1.2,        # ft
    "release_extension": 1.0,  # ft
}

K_RV_P, K_RV_B = 250.0, 300.0
K_WHIFF_P, K_WHIFF_B = 40.0, 60.0
K_CHASE_P, K_CHASE_B = 60.0, 80.0
K_XW_P, K_XW_B = 40.0, 50.0
MIN_USAGE = 0.03
POSITION_PLAYER_VELO = 75.0  # avg mph below this = position player pitching

# Data loading/preparation
def available_years() -> list[int]:
    return sorted(int(p.stem.split("_")[1]) for p in DATA_DIR.glob("statcast_*.parquet"))


def load_season(year: int) -> tuple[pd.DataFrame, dict[int, str]]:
    df = pd.read_parquet(DATA_DIR / f"statcast_{year}.parquet")
    names = pd.read_parquet(DATA_DIR / f"players_{year}.parquet")
    name_map = dict(zip(names["player_id"].astype(int), names["name"]))
    return prepare(df), name_map


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    df = df[df["pitch_type"].notna() & ~df["pitch_type"].isin(EXCLUDE_TYPES)].copy()
    for c in df.columns:
        if pd.api.types.is_numeric_dtype(df[c]) and c not in ("batter", "pitcher", "game_pk"):
            df[c] = df[c].astype("float64")
        elif c in ("pitch_type", "stand", "p_throws", "description", "events", "bb_type",
                   "inning_topbot", "home_team", "away_team", "game_type", "type"):
            df[c] = df[c].astype(object).where(df[c].notna(), None)
    df["game_pk"] = df["game_pk"].astype("int64")
    df = df[df["release_speed"].notna() & df["pfx_x"].notna() & df["pfx_z"].notna()]
    df["batter"] = df["batter"].astype(int)
    df["pitcher"] = df["pitcher"].astype(int)

    arm = np.where(df["p_throws"] == "R", -1.0, 1.0)  # mirror lefties onto arm side
    df["ivb"] = df["pfx_z"] * 12.0
    df["hb_arm"] = df["pfx_x"] * 12.0 * arm
    df["hb"] = df["pfx_x"] * 12.0
    df["rel_x_arm"] = df["release_pos_x"] * arm
    for c in ["release_spin_rate", "release_extension", "release_pos_z", "rel_x_arm"]:
        df[c] = df[c].fillna(df.groupby(["pitcher", "pitch_type"])[c].transform("median"))
        df[c] = df[c].fillna(df[c].median())

    desc = df["description"].astype(str)
    df["is_swing"] = desc.isin(SWING_DESC).astype(np.int8)
    df["is_whiff"] = desc.isin(WHIFF_DESC).astype(np.int8)
    df["is_csw"] = (desc.isin(WHIFF_DESC) | desc.isin(CALLED_STRIKE)).astype(np.int8)
    df["in_zone"] = df["zone"].between(1, 9).astype(np.int8)
    df["is_ooz"] = (df["zone"] > 9).astype(np.int8)
    df["is_chase"] = (df["is_ooz"].astype(bool) & df["is_swing"].astype(bool)).astype(np.int8)
    df["is_bip"] = (desc == "hit_into_play").astype(np.int8)
    xw = df["estimated_woba_using_speedangle"]
    df["xwoba_con"] = np.where(df["is_bip"] == 1, xw.fillna(df["woba_value"]), np.nan)
    df["hard_hit"] = np.where(df["is_bip"] == 1, (df["launch_speed"] >= 95).astype(float), np.nan)

    # Count-neutral run value delta RE minus the league average for that count
    rv = df["delta_run_exp"].fillna(0.0)
    df["rv"] = rv - rv.groupby([df["balls"], df["strikes"]]).transform("mean")

    df["bat_team"] = np.where(df["inning_topbot"] == "Top", df["away_team"], df["home_team"])
    df["pit_team"] = np.where(df["inning_topbot"] == "Top", df["home_team"], df["away_team"])
    df["game_date"] = pd.to_datetime(df["game_date"])
    df["pitch_name"] = df["pitch_type"].map(PITCH_NAMES).fillna(df["pitch_type"])
    return df.reset_index(drop=True)


# Engine
@dataclass
class PitchEstimate:
    pitch_type: str
    usage: float
    n_pitcher: int          # pitcher's pitches of this type vs this side
    n_eff_batter: float     # effective # of similar pitches the batter has seen
    shape: dict
    baseline: dict 
    pitcher: dict         
    batter_on_similar: dict 
    expected: dict
    edge_rv100: float


@dataclass
class MatchupResult:
    batter: int
    pitcher: int
    stand: str
    p_throws: str
    pitches: list[PitchEstimate] = field(default_factory=list)

    @property
    def rv100(self) -> float:
        return float(sum(p.usage * p.expected["rv"] for p in self.pitches) * 100)

    def weighted(self, key: str, src: str = "expected") -> float:
        vals = [(p.usage, getattr(p, src)[key]) for p in self.pitches if not np.isnan(getattr(p, src)[key])]
        if not vals:
            return np.nan
        w = sum(u for u, _ in vals)
        return float(sum(u * v for u, v in vals) / w)

    @property
    def confidence(self) -> float:
        """0-1: how much real batter evidence underlies the estimate."""
        n = sum(p.usage * p.n_eff_batter for p in self.pitches)
        return float(n / (n + K_RV_B))


def _logit(p):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def _expit(x):
    return 1.0 / (1.0 + np.exp(-x))


class MatchupEngine:
    METRICS = ("rv", "whiff", "chase", "xwoba_con")

    def __init__(self, df: pd.DataFrame, names: dict[int, str]):
        self.df = df
        self.names = names
        self.feat = list(KERNEL_FEATURES)
        self.bw = np.array(list(KERNEL_FEATURES.values()))

        # League pools keyed by (p_throws, stand)
        self.pools = {}
        for key, g in df.groupby(["p_throws", "stand"]):
            self.pools[key] = self._arrays(g)
        self.by_batter = {k: v for k, v in df.groupby("batter").groups.items()}
        self.by_pitcher = {k: v for k, v in df.groupby("pitcher").groups.items()}
        self._baseline_cache: dict = {}
        # Position players hidden from pitcher pickers and staff lists.
        velo = df.groupby("pitcher")["release_speed"].mean()
        self.position_pitchers = set(velo[velo < POSITION_PLAYER_VELO].index)

    # helpers
    def _arrays(self, g: pd.DataFrame) -> dict:
        return {
            "X": (g[self.feat].to_numpy(float) / self.bw),
            "rv": g["rv"].to_numpy(float),
            "swing": g["is_swing"].to_numpy(float),
            "whiff": g["is_whiff"].to_numpy(float),
            "ooz": g["is_ooz"].to_numpy(float),
            "chase": g["is_chase"].to_numpy(float),
            "bip": g["is_bip"].to_numpy(float),
            "xw": np.nan_to_num(g["xwoba_con"].to_numpy(float)),
        }

    @staticmethod
    def _kernel_sums(A: dict, c: np.ndarray) -> dict:
        d2 = ((A["X"] - c) ** 2).sum(axis=1)
        w = np.exp(-0.5 * d2)
        return {
            "w": w.sum(),
            "rv": (w * A["rv"]).sum(),
            "swing": (w * A["swing"]).sum(),
            "whiff": (w * A["whiff"]).sum(),
            "ooz": (w * A["ooz"]).sum(),
            "chase": (w * A["chase"]).sum(),
            "bip": (w * A["bip"]).sum(),
            "xw": (w * A["xw"]).sum(),
        }

    @staticmethod
    def _rates(s: dict) -> dict:
        def r(a, b):
            return a / b if b > 0 else np.nan
        return {
            "rv": r(s["rv"], s["w"]),
            "whiff": r(s["whiff"], s["swing"]),
            "chase": r(s["chase"], s["ooz"]),
            "xwoba_con": r(s["xw"], s["bip"]),
        }

    @staticmethod
    def _shrink(s: dict, prior: dict, k_rv, k_wh, k_ch, k_xw) -> dict:
        return {
            "rv": (s["rv"] + k_rv * prior["rv"]) / (s["w"] + k_rv),
            "whiff": (s["whiff"] + k_wh * prior["whiff"]) / (s["swing"] + k_wh),
            "chase": (s["chase"] + k_ch * prior["chase"]) / (s["ooz"] + k_ch),
            "xwoba_con": (s["xw"] + k_xw * prior["xwoba_con"]) / (s["bip"] + k_xw),
        }

    def name(self, pid: int) -> str:
        return self.names.get(int(pid), str(pid))

    def batter_df(self, bid: int) -> pd.DataFrame:
        return self.df.loc[self.by_batter.get(bid, [])]

    def pitcher_df(self, pid: int) -> pd.DataFrame:
        return self.df.loc[self.by_pitcher.get(pid, [])]

    def pitcher_hand(self, pid: int) -> str:
        return self.pitcher_df(pid)["p_throws"].mode().iat[0]

    def batter_stand_vs(self, bid: int, p_throws: str) -> str:
        b = self.batter_df(bid)
        s = b.loc[b["p_throws"] == p_throws, "stand"]
        if s.empty:
            s = b["stand"]
            if s.nunique() > 1:  # switch hitter with no data vs this hand
                return "L" if p_throws == "R" else "R"
        return s.mode().iat[0]

    def arsenal(self, pid: int, stand: str | None = None) -> pd.DataFrame:
        """Per-pitch-type arsenal table (optionally vs one batter side)."""
        p = self.pitcher_df(pid)
        if stand:
            p = p[p["stand"] == stand]
        if p.empty:
            return pd.DataFrame()
        g = p.groupby("pitch_type")
        out = pd.DataFrame({
            "Pitch": g.size().index.map(lambda t: PITCH_NAMES.get(t, t)),
            "N": g.size(),
            "Usage": g.size() / len(p),
            "Velo": g["release_speed"].mean(),
            "Spin": g["release_spin_rate"].mean(),
            "IVB": g["ivb"].mean(),
            "HB (arm+)": g["hb_arm"].mean(),
            "Ext": g["release_extension"].mean(),
            "Whiff%": g["is_whiff"].sum() / g["is_swing"].sum().replace(0, np.nan),
            "CSW%": g["is_csw"].mean(),
            "Chase%": g["is_chase"].sum() / g["is_ooz"].sum().replace(0, np.nan),
            "xwOBAcon": g["xwoba_con"].mean(),
            "RV/100 (pit)": -g["rv"].mean() * 100,
        })
        return out.sort_values("Usage", ascending=False)

    def batter_vs_types(self, bid: int, p_throws: str | None = None) -> pd.DataFrame:
        b = self.batter_df(bid)
        if p_throws:
            b = b[b["p_throws"] == p_throws]
        if b.empty:
            return pd.DataFrame()
        g = b.groupby("pitch_type")
        out = pd.DataFrame({
            "Pitch": g.size().index.map(lambda t: PITCH_NAMES.get(t, t)),
            "Group": g.size().index.map(lambda t: PITCH_GROUP.get(t, "Other")),
            "Seen": g.size(),
            "Swing%": g["is_swing"].mean(),
            "Whiff%": g["is_whiff"].sum() / g["is_swing"].sum().replace(0, np.nan),
            "Chase%": g["is_chase"].sum() / g["is_ooz"].sum().replace(0, np.nan),
            "xwOBAcon": g["xwoba_con"].mean(),
            "HardHit%": g["hard_hit"].mean(),
            "EV": g["launch_speed"].mean().where(g["is_bip"].sum() > 0),
            "RV/100 (bat)": g["rv"].mean() * 100,
        })
        return out.sort_values("Seen", ascending=False)

    def head_to_head(self, bid: int, pid: int) -> pd.DataFrame:
        d = self.df[(self.df["batter"] == bid) & (self.df["pitcher"] == pid)]
        return d.sort_values(["game_date", "at_bat_number", "pitch_number"])

    # model
    def _baseline(self, p_throws: str, stand: str, c: np.ndarray) -> tuple[dict, dict]:
        key = (p_throws, stand, tuple(np.round(c, 3)))
        if key not in self._baseline_cache:
            s = self._kernel_sums(self.pools[(p_throws, stand)], c)
            self._baseline_cache[key] = (s, self._rates(s))
        return self._baseline_cache[key]

    @lru_cache(maxsize=4096)
    def _pitcher_profile(self, pid: int, stand: str):
        """Per pitch type: usage, centroid, baseline, shrunk pitcher estimate."""
        p_all = self.pitcher_df(pid)
        if p_all.empty:
            return p_throws_none()
        p_throws = p_all["p_throws"].mode().iat[0]
        p_side = p_all[p_all["stand"] == stand]
        # Usage vs this side, fall back to overall if very small sample.
        src = p_side if len(p_side) >= 100 else p_all
        usage = src["pitch_type"].value_counts(normalize=True)
        rows = []
        for pt, u in usage.items():
            if u < MIN_USAGE:
                continue
            shape_df = p_all[p_all["pitch_type"] == pt]
            c = shape_df[self.feat].mean().to_numpy(float) / self.bw
            _, base = self._baseline(p_throws, stand, c)
            side_df = p_side[p_side["pitch_type"] == pt]
            s = self._kernel_sums_exact(side_df)
            est = self._shrink(s, base, K_RV_P, K_WHIFF_P, K_CHASE_P, K_XW_P)
            shape = {
                "velo": shape_df["release_speed"].mean(),
                "spin": shape_df["release_spin_rate"].mean(),
                "ivb": shape_df["ivb"].mean(),
                "hb": shape_df["hb_arm"].mean(),
                "ext": shape_df["release_extension"].mean(),
                "rel_z": shape_df["release_pos_z"].mean(),
            }
            rows.append((pt, u, len(side_df), c, shape, base, est))
        tot = sum(r[1] for r in rows)
        rows = [(pt, u / tot, n, c, sh, b, e) for pt, u, n, c, sh, b, e in rows]
        return p_throws, rows

    @staticmethod
    def _kernel_sums_exact(g: pd.DataFrame) -> dict:
        """Unweighted sums for a pitcher's own pitches of one type."""
        return {
            "w": float(len(g)),
            "rv": g["rv"].sum(),
            "swing": g["is_swing"].sum(),
            "whiff": g["is_whiff"].sum(),
            "ooz": g["is_ooz"].sum(),
            "chase": g["is_chase"].sum(),
            "bip": g["is_bip"].sum(),
            "xw": g["xwoba_con"].fillna(0).sum(),
        }

    @lru_cache(maxsize=8192)
    def _batter_pool(self, bid: int, p_throws: str):
        b = self.batter_df(bid)
        b = b[b["p_throws"] == p_throws]
        return self._arrays(b) if len(b) else None

    def matchup(self, bid: int, pid: int) -> MatchupResult:
        p_throws = self.pitcher_hand(pid)
        stand = self.batter_stand_vs(bid, p_throws)
        _, rows = self._pitcher_profile(pid, stand)
        bpool = self._batter_pool(bid, p_throws)
        res = MatchupResult(bid, pid, stand, p_throws)
        for pt, u, n, c, shape, base, pest in rows:
            if bpool is not None:
                bs = self._kernel_sums(bpool, c)
                bsh = self._shrink(bs, base, K_RV_B, K_WHIFF_B, K_CHASE_B, K_XW_B)
                neff = bs["w"]
            else:
                bsh, neff = dict(base), 0.0
            exp = {
                "rv": pest["rv"] + (bsh["rv"] - base["rv"]),
                "whiff": float(_expit(_logit(pest["whiff"]) + _logit(bsh["whiff"]) - _logit(base["whiff"]))),
                "chase": float(_expit(_logit(pest["chase"]) + _logit(bsh["chase"]) - _logit(base["chase"]))),
                "xwoba_con": pest["xwoba_con"] + (bsh["xwoba_con"] - base["xwoba_con"]),
            }
            res.pitches.append(PitchEstimate(
                pitch_type=pt, usage=u, n_pitcher=n, n_eff_batter=neff, shape=shape,
                baseline=base, pitcher=pest, batter_on_similar=bsh, expected=exp,
                edge_rv100=exp["rv"] * 100,
            ))
        return res

    def teams(self) -> list[str]:
        return sorted(self.df["bat_team"].unique())

    def roster(self, team: str, role: str, min_n: int, last_n_days: int | None = None) -> pd.DataFrame:
        """Players whose most recent appearance was for `team`."""
        col, team_col = ("batter", "bat_team") if role == "batter" else ("pitcher", "pit_team")
        last = self.df.sort_values("game_date").groupby(col).tail(1)[[col, team_col, "game_date"]]
        ids = last.loc[last[team_col] == team, col]
        if role == "pitcher":
            ids = ids[~ids.isin(self.position_pitchers)]
        d = self.df[self.df[col].isin(ids)]
        if role == "batter":
            cnt = d.groupby(col).apply(lambda g: g.groupby(["game_pk", "at_bat_number"]).ngroups)
            label = "PA"
        else:
            cnt = d.groupby(col).size()
            label = "Pitches"
        out = cnt.rename(label).reset_index().rename(columns={col: "id"})
        out = out[out[label] >= min_n].sort_values(label, ascending=False)
        out["name"] = out["id"].map(self.name)
        if role == "pitcher":
            out["hand"] = out["id"].map(self.pitcher_hand)
            # starter = whoever threw his team's first pitch of the game
            team_df = self.df[self.df["pit_team"] == team]
            first = team_df.loc[team_df.groupby("game_pk")["at_bat_number"].idxmin()]
            gs = first["pitcher"].value_counts()
            out["GS"] = out["id"].map(gs).fillna(0).astype(int)
            out["role"] = np.where(out["GS"] >= 3, "SP", "RP")
        return out.reset_index(drop=True)

    def grid(self, batters: list[int], pitchers: list[int]) -> pd.DataFrame:
        recs = []
        for b in batters:
            for p in pitchers:
                m = self.matchup(b, p)
                best = max(m.pitches, key=lambda x: -x.expected["rv"]) if m.pitches else None
                worst = max(m.pitches, key=lambda x: x.expected["rv"]) if m.pitches else None
                recs.append({
                    "batter": b, "pitcher": p,
                    "Batter": self.name(b), "Pitcher": self.name(p),
                    "rv100": m.rv100, "whiff": m.weighted("whiff"),
                    "xwoba_con": m.weighted("xwoba_con"), "chase": m.weighted("chase"),
                    "confidence": m.confidence,
                    "pitcher_best": PITCH_NAMES.get(best.pitch_type, best.pitch_type) if best else "",
                    "batter_best": PITCH_NAMES.get(worst.pitch_type, worst.pitch_type) if worst else "",
                })
        return pd.DataFrame(recs)

    XS = np.linspace(-1.6, 1.6, 17)
    ZS = np.linspace(0.8, 4.2, 18)

    def batter_damage_map(self, bid: int, p_throws: str, bw: float = 0.3, k: float = 25.0) -> np.ndarray:
        b = self.batter_df(bid)
        b = b[(b["p_throws"] == p_throws) & b["plate_x"].notna()]
        gx, gz = np.meshgrid(self.XS, self.ZS)
        if b.empty:
            return np.zeros_like(gx)
        px, pz, rv = b["plate_x"].to_numpy(), b["plate_z"].to_numpy(), b["rv"].to_numpy()
        d2 = ((gx[..., None] - px) ** 2 + (gz[..., None] - pz) ** 2) / bw ** 2
        w = np.exp(-0.5 * d2)
        return (w * rv).sum(-1) / (w.sum(-1) + k) * 100

    def pitcher_location_map(self, pid: int, stand: str, pitch_type: str | None = None, bw: float = 0.3) -> np.ndarray:
        p = self.pitcher_df(pid)
        p = p[(p["stand"] == stand) & p["plate_x"].notna()]
        if pitch_type:
            p = p[p["pitch_type"] == pitch_type]
        gx, gz = np.meshgrid(self.XS, self.ZS)
        if p.empty:
            return np.zeros_like(gx)
        px, pz = p["plate_x"].to_numpy(), p["plate_z"].to_numpy()
        d2 = ((gx[..., None] - px) ** 2 + (gz[..., None] - pz) ** 2) / bw ** 2
        dens = np.exp(-0.5 * d2).sum(-1)
        return dens / dens.sum()

    def location_edge(self, bid: int, pid: int) -> float:
        """Where this pitcher throws x where this batter does damage (RV/100, batter view)."""
        p_throws = self.pitcher_hand(pid)
        stand = self.batter_stand_vs(bid, p_throws)
        return float((self.pitcher_location_map(pid, stand) * self.batter_damage_map(bid, p_throws)).sum())

    # movement space
    HBS = np.linspace(-22, 24, 24)   # arm-side break (in)
    IVBS = np.linspace(-20, 22, 22)  # induced vertical break (in)

    def batter_movement_map(self, bid: int, p_throws: str, metric: str = "rv",
                            bw: float = 3.5, k: float = 30.0) -> np.ndarray:
        """Batter outcome by pitch movement (arm-side HB x IVB), shrunk to 0 / league."""
        b = self.batter_df(bid)
        b = b[b["p_throws"] == p_throws]
        gh, gv = np.meshgrid(self.HBS, self.IVBS)
        if b.empty:
            return np.full_like(gh, np.nan)
        hb, iv = b["hb_arm"].to_numpy(), b["ivb"].to_numpy()
        w = np.exp(-0.5 * (((gh[..., None] - hb) ** 2 + (gv[..., None] - iv) ** 2) / bw ** 2))
        if metric == "rv":
            return (w * b["rv"].to_numpy()).sum(-1) / (w.sum(-1) + k) * 100
        if metric == "whiff":
            sw = b["is_swing"].to_numpy()
            lg = self.df["is_whiff"].sum() / self.df["is_swing"].sum()
            return ((w * b["is_whiff"].to_numpy()).sum(-1) + 15 * lg) / ((w * sw).sum(-1) + 15)
        raise ValueError(metric)

    # directory
    @lru_cache(maxsize=1)
    def directory(self) -> pd.DataFrame:
        """All players with their latest team, as hitter and as pitcher."""
        d = self.df.sort_values("game_date")
        bat = d.groupby("batter").agg(team=("bat_team", "last"), n=("batter", "size")).reset_index()
        bat = bat.rename(columns={"batter": "id"}).assign(role="batter")
        pit = d.groupby("pitcher").agg(team=("pit_team", "last"), n=("pitcher", "size")).reset_index()
        pit = pit.rename(columns={"pitcher": "id"}).assign(role="pitcher")
        pit = pit[~pit["id"].isin(self.position_pitchers)]
        out = pd.concat([bat, pit], ignore_index=True)
        out["name"] = out["id"].map(self.name)
        out["label"] = out["name"] + " (" + out["team"] + ")"
        return out.sort_values("n", ascending=False)


def p_throws_none():
    return None, []


# Plain-language scouting notes
def edge_label(rv100: float) -> str:
    a = abs(rv100)
    if a < 0.4:
        return "Even"
    side = "Batter" if rv100 > 0 else "Pitcher"
    strength = "slight" if a < 1.2 else "moderate" if a < 2.5 else "strong"
    return f"{side} — {strength}"


def scouting_notes(eng: MatchupEngine, m: MatchupResult) -> tuple[list[str], list[str]]:
    """Returns (batter advantages, pitcher advantages) as bullet strings."""
    bat, pit = [], []
    for p in sorted(m.pitches, key=lambda x: -x.usage):
        nm = PITCH_NAMES.get(p.pitch_type, p.pitch_type)
        tag = f"**{nm}** ({p.usage:.0%}, {p.shape['velo']:.1f} mph, {p.shape['ivb']:+.0f}\" IVB, {p.shape['hb']:+.0f}\" HB, {p.shape['spin']:.0f} rpm)"
        b_eff = (p.batter_on_similar["rv"] - p.baseline["rv"]) * 100
        p_eff = (p.pitcher["rv"] - p.baseline["rv"]) * 100
        rel = "" if p.n_eff_batter >= 60 else " _(thin sample)_"
        if b_eff >= 0.6:
            bat.append(f"{tag}: hitter is {b_eff:+.1f} RV/100 better than average vs pitches shaped like this "
                       f"(~{p.n_eff_batter:.0f} similar pitches seen){rel}.")
        elif b_eff <= -0.6:
            pit.append(f"{tag}: hitter struggles with this shape, {b_eff:+.1f} RV/100 vs average "
                       f"(~{p.n_eff_batter:.0f} similar pitches seen){rel}.")
        if p_eff <= -0.8:
            pit.append(f"{tag}: plays above its stuff vs {'LHH' if m.stand == 'L' else 'RHH'} "
                       f"({-p_eff:.1f} RV/100 better than league on this shape).")
        elif p_eff >= 0.8:
            bat.append(f"{tag}: gets hit harder than its shape suggests vs {'LHH' if m.stand == 'L' else 'RHH'} "
                       f"({p_eff:+.1f} RV/100).")
        wd = p.expected["whiff"] - p.baseline["whiff"]
        if p.usage >= 0.08 and wd >= 0.06:
            pit.append(f"{tag}: projected whiff rate {p.expected['whiff']:.0%} vs league {p.baseline['whiff']:.0%} on this shape.")
        elif p.usage >= 0.08 and wd <= -0.06:
            bat.append(f"{tag}: hitter makes contact — projected whiff {p.expected['whiff']:.0%} vs league {p.baseline['whiff']:.0%}.")
        xd = p.expected["xwoba_con"] - p.baseline["xwoba_con"]
        if p.usage >= 0.08 and xd >= 0.04:
            bat.append(f"{tag}: projected xwOBA on contact {p.expected['xwoba_con']:.3f} (league {p.baseline['xwoba_con']:.3f}).")
        elif p.usage >= 0.08 and xd <= -0.04:
            pit.append(f"{tag}: suppresses contact quality — projected xwOBAcon {p.expected['xwoba_con']:.3f} (league {p.baseline['xwoba_con']:.3f}).")
    if m.p_throws != m.stand:
        bat.append(f"Platoon: {'LHH' if m.stand == 'L' else 'RHH'} vs {'LHP' if m.p_throws == 'L' else 'RHP'} (opposite-hand).")
    else:
        pit.append(f"Platoon: same-handed matchup ({m.stand}HH vs {m.p_throws}HP).")
    return bat, pit
