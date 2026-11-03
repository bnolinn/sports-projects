import unicodedata
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pybaseball import statcast_batter, statcast_pitcher, playerid_lookup, cache

cache.enable()

BAT_START = "2021-03-01"    # how far back to pull hitters (longer = slower)
PITCH_START = "2026-03-01"  # pitcher profile = this season's pitches
END = "2026-10-19"
FEATURES = ["release_speed", "release_spin_rate", "pfx_x", "pfx_z"]
K = 30           # how many look-alike pitches to use per hitter
MIN_POOL = 25    # skip if hitter has fewer PA vs that pitch type and hand
MIN_USAGE = 0.05 # ignore pitch types a pitcher throws under 5% of the time
SHRINK_PA = 30   # strength of shrinkage toward the hitter's overall result
# Sweepers and slurves are grouped with sliders and curves; the movement
# matching below still separates them by shape.
GROUP = {"ST": "SL", "SV": "SL", "KC": "CU"}

RAYS_BATTERS = [
    ("Liam", "Hicks"), ("Nick", "Fortes"), ("Jonathan", "Aranda"),
    ("Richie", "Palacios"), ("Ben", "Williamson"), ("Taylor", "Walls"),
    ("Jorge", "Mateo"), ("Junior", "Caminero"), ("Chandler", "Simpson"),
    ("Cedric", "Mullins"), ("Jonny", "DeLuca"), ("Victor", "Mesa"),
    ("Ryan", "Vilade"), ("Yandy", "Diaz"),
]
RAYS_PITCHERS = [
    ("Griffin", "Jax"), ("Nick", "Martinez"), ("Steven", "Matz"),
    ("Freddy", "Peralta"), ("Drew", "Rasmussen"), ("William", "Hopkins"),
    ("Bryan", "Baker"), ("Kevin", "Kelly"), ("Ian", "Seymour"),
    ("Cam", "Booser"), ("Tyler", "Wells"), ("Cole", "Sulser"),
]
BREWERS_BATTERS = [
    ("William", "Contreras"), ("Gary", "Sanchez"), ("Jake", "Bauers"),
    ("Andrew", "Vaughn"), ("Brice", "Turang"), ("Joey", "Ortiz"),
    ("Cooper", "Pratt"), ("David", "Hamilton"), ("Jackson", "Chourio"),
    ("Garrett", "Mitchell"), ("Sal", "Frelick"), ("Luis", "Lara"),
    ("Christian", "Yelich"),
]
BREWERS_PITCHERS = [
    ("Dustin", "May"), ("Jacob", "Misiorowski"), ("Logan", "Henderson"),
    ("Kyle", "Harrison"), ("Robert", "Gasser"), ("Aaron", "Ashby"),
    ("Trevor", "Megill"), ("JoJo", "Romero"), ("DL", "Hall"),
    ("Chad", "Patrick"), ("Abner", "Uribe"), ("Antonio", "Senzatela"),
    ("Shane", "Drohan"),
]


def strip_accents(s):
    return "".join(c for c in unicodedata.normalize("NFD", str(s))
                   if unicodedata.category(c) != "Mn").lower()


def get_id(first, last):
    res = playerid_lookup(last, first, fuzzy=True)
    if res.empty:
        print(f"!! Not found: {first} {last}")
        return None
    res = res.copy()
    res["mlb_played_last"] = pd.to_numeric(res["mlb_played_last"], errors="coerce")
    res = res[res["key_mlbam"].notna()]
    last_n = res["name_last"].map(strip_accents)
    first_n = res["name_first"].map(strip_accents)
    exact = res[(first_n == strip_accents(first)) & (last_n == strip_accents(last))]
    if exact.empty:
        recent = res[(last_n == strip_accents(last)) & (res["mlb_played_last"] >= 2024)]
        if len(recent) == 1:
            row = recent.iloc[0]
            print(f"~ Using {row['name_first']} {row['name_last']} for {first} {last}")
            return int(row["key_mlbam"])
        print(f"!! No exact match for {first} {last}. Closest:")
        print(res[["name_first", "name_last", "mlb_played_last"]].to_string(index=False))
        return None
    exact = exact.sort_values("mlb_played_last", ascending=False)
    return int(exact.iloc[0]["key_mlbam"])


def build_ids(players):
    out = {}
    for first, last in players:
        pid = get_id(first, last)
        if pid:
            out[pid] = f"{first} {last}"
    return out


rays_b = build_ids(RAYS_BATTERS)
rays_p = build_ids(RAYS_PITCHERS)
brew_b = build_ids(BREWERS_BATTERS)
brew_p = build_ids(BREWERS_PITCHERS)


def prep_pa(df):
    pa = df[df["events"].notna()].copy()
    pa = pa[pa["woba_denom"].fillna(0) == 1]  # drops bunts, IBB, etc.
    pa["xw"] = pa["estimated_woba_using_speedangle"].fillna(pa["woba_value"])
    pa["pt"] = pa["pitch_type"].replace(GROUP)
    return pa.dropna(subset=FEATURES + ["pt", "p_throws", "xw"])


bat_pa = {}
for pid, name in {**rays_b, **brew_b}.items():
    print(f"Pulling hitter {name}...")
    try:
        df = statcast_batter(BAT_START, END, pid)
    except Exception as e:
        print(f"!! Failed for {name}: {e}")
        continue
    if df is not None and not df.empty:
        bat_pa[name] = prep_pa(df)

def pitcher_profile(df):
    d = df.copy()
    d["pt"] = d["pitch_type"].replace(GROUP)
    d = d.dropna(subset=FEATURES + ["pt"])
    if d.empty:
        return None
    hand = d["p_throws"].mode()[0]
    prof = d.groupby("pt").agg(
        n=("pt", "size"),
        **{f: (f, "mean") for f in FEATURES},
    ).reset_index()
    prof["usage"] = prof["n"] / prof["n"].sum()
    prof = prof[prof["usage"] >= MIN_USAGE].copy()
    prof["usage"] = prof["usage"] / prof["usage"].sum()
    return hand, prof


profiles = {}
for pid, name in {**rays_p, **brew_p}.items():
    print(f"Pulling pitcher {name}...")
    try:
        df = statcast_pitcher(PITCH_START, END, pid)
    except Exception as e:
        print(f"!! Failed for {name}: {e}")
        continue
    if df is None or df.empty:
        print(f"!! No 2026 data for {name}")
        continue
    p = pitcher_profile(df)
    if p:
        profiles[name] = p

def compute(batter_names, pitcher_names):
    rows = []
    for b in batter_names:
        pa = bat_pa.get(b)
        if pa is None or pa.empty:
            continue
        for p in pitcher_names:
            if p not in profiles:
                continue
            hand, prof = profiles[p]
            for _, r in prof.iterrows():
                pool = pa[(pa["pt"] == r["pt"]) & (pa["p_throws"] == hand)]
                if len(pool) < MIN_POOL:
                    continue
                X = pool[FEATURES].to_numpy(float)
                sd = X.std(axis=0)
                sd[sd == 0] = 1
                target = r[FEATURES].to_numpy(float)
                dist = np.sqrt((((X - target) / sd) ** 2).sum(axis=1))
                idx = np.argsort(dist)[:K]
                matched = pool["xw"].to_numpy()[idx].mean()
                prior = pool["xw"].mean()
                n = len(idx)
                shrunk = (n * matched + SHRINK_PA * prior) / (n + SHRINK_PA)
                rows.append(dict(batter=b, pitcher=p, pitch=r["pt"],
                                 usage=round(r["usage"], 3), n_matched=n,
                                 avg_dist=round(dist[idx].mean(), 2),
                                 raw=round(matched, 3), prior=round(prior, 3),
                                 shrunk=round(shrunk, 3)))
    return pd.DataFrame(rows)


def rollup(detail):
    d = detail.copy()
    d["wn"] = d["usage"] / d.groupby(["batter", "pitcher"])["usage"].transform("sum")
    d["s"] = d["wn"] * d["shrunk"]
    d["r"] = d["wn"] * d["raw"]
    out = d.groupby(["batter", "pitcher"]).agg(
        xwOBA_shrunk=("s", "sum"), xwOBA_raw=("r", "sum"),
        coverage=("usage", "sum")).reset_index()
    return out.round(3)


brew_detail = compute(brew_b.values(), rays_p.values())   # Brewers hitters vs Rays arms
rays_detail = compute(rays_b.values(), brew_p.values())   # Rays hitters vs Brewers arms
brew_roll = rollup(brew_detail) if not brew_detail.empty else pd.DataFrame()
rays_roll = rollup(rays_detail) if not rays_detail.empty else pd.DataFrame()

def show(detail, batter, pitcher):
    sub = detail[(detail["batter"] == batter) & (detail["pitcher"] == pitcher)]
    print(sub.drop(columns=["batter", "pitcher"]).to_string(index=False))

def heatmap(roll, title):
    if roll.empty:
        print(f"No data for: {title}")
        return
    piv = roll.pivot(index="batter", columns="pitcher", values="xwOBA_shrunk")
    fig, ax = plt.subplots(figsize=(max(6, 0.9 * piv.shape[1] + 3),
                                    max(4, 0.6 * piv.shape[0] + 2)))
    im = ax.imshow(piv.values.astype(float), cmap="RdBu_r", vmin=0.25, vmax=0.42)
    ax.set_xticks(range(len(piv.columns)))
    ax.set_xticklabels(piv.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(piv.index)))
    ax.set_yticklabels(piv.index)
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            v = piv.iloc[i, j]
            if pd.notna(v):
                ax.text(j, i, f"{v:.3f}", ha="center", va="center", fontsize=7)
    ax.set_title(title)
    fig.colorbar(im, label="Expected xwOBA (red = hitter wins)")
    plt.tight_layout()
    plt.show()


def pitcher_bars(roll, title):
    if roll.empty:
        return
    s = roll.groupby("pitcher")["xwOBA_shrunk"].mean().sort_values()
    fig, ax = plt.subplots(figsize=(7, max(3, 0.4 * len(s) + 1.5)))
    ax.barh(s.index, s.values)
    ax.axvline(s.mean(), linestyle="--", color="gray")
    ax.set_xlabel("Avg expected xwOBA allowed to opposing lineup (lower = better)")
    ax.set_title(title)
    plt.tight_layout()
    plt.show()


heatmap(brew_roll, "Brewers hitters vs Rays pitchers: pitch-shape matched xwOBA")
heatmap(rays_roll, "Rays hitters vs Brewers pitchers: pitch-shape matched xwOBA")
pitcher_bars(brew_roll, "Rays pitchers vs Brewers lineup")
pitcher_bars(rays_roll, "Brewers pitchers vs Rays lineup")

print("\nRays pitchers vs Brewers lineup")
print(brew_roll.sort_values("xwOBA_shrunk").to_string(index=False))
print("\nBrewers pitchers vs Rays lineup")
print(rays_roll.sort_values("xwOBA_shrunk").to_string(index=False))