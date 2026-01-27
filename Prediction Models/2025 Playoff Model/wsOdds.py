import pandas as pd
import numpy as np
import time
import requests
from pybaseball import team_batting, team_pitching, team_fielding

def nickname_from_full(name: str) -> str:
    if not isinstance(name, str) or not name:
        return name
    parts = name.split()
    if parts[-1] in {"Sox", "Jays"} and len(parts) >= 2:
        return " ".join(parts[-2:])
    return parts[-1]

def clean_team_string(s: pd.Series) -> pd.Series:
    return (
        s.astype(str)
         .str.replace('\u00a0', ' ', regex=False) 
         .str.replace(r'\s+', ' ', regex=True)    
         .str.strip()
    )

def pyth_win_pct(rs, ra, exponent: float = 1.83) -> pd.Series:
    rs = pd.to_numeric(rs, errors='coerce')
    ra = pd.to_numeric(ra, errors='coerce')
    num = rs.pow(exponent)
    den = num + ra.pow(exponent)
    return round((num / den), 3)

def fetch_mlb_standings_runs(year: int, max_retries: int = 3, sleep_sec: float = 1.0) -> pd.DataFrame:
    url = f"https://statsapi.mlb.com/api/v1/standings?leagueId=103,104&season={year}&standingsTypes=regularSeason"
    tries = 0
    while tries < max_retries:
        resp = requests.get(url, timeout=30)
        if resp.status_code == 200:
            data = resp.json()
            rows = []
            for record in data.get('records', []):
                for tr in record.get('teamRecords', []):
                    team_name = tr.get('team', {}).get('name')
                    if not team_name:
                        continue
                    rows.append({
                        'Year': year,
                        'Team_Full': team_name,
                        'Wins_mlb': tr.get('wins'),
                        'Losses_mlb': tr.get('losses'),
                        'Win_Pct_mlb': pd.to_numeric(tr.get('winningPercentage'), errors='coerce'),
                        'GB_mlb': tr.get('gamesBack', '0'),
                        'WCGB_mlb': tr.get('wildCardGamesBack', '0'),
                        'Streak_mlb': tr.get('streak', {}).get('streakCode', 'N/A'),
                        'R_mlb': tr.get('runsScored'),
                        'RA_mlb': tr.get('runsAllowed')
                    })
            df = pd.DataFrame(rows)
            if not df.empty:
                df['Team_Full'] = clean_team_string(df['Team_Full'])
                df['Team_Nickname'] = df['Team_Full'].apply(nickname_from_full)
                for c in ['Wins_mlb', 'Losses_mlb', 'R_mlb', 'RA_mlb']:
                    df[c] = pd.to_numeric(df[c], errors='coerce').astype('Int64')
                return df
            return pd.DataFrame(columns=[
                'Year','Team_Full','Team_Nickname','Wins_mlb','Losses_mlb',
                'Win_Pct_mlb','GB_mlb','WCGB_mlb','Streak_mlb','R_mlb','RA_mlb'
            ])
        tries += 1
        time.sleep(sleep_sec * (2 ** tries))
    return pd.DataFrame(columns=[
        'Year','Team_Full','Team_Nickname','Wins_mlb','Losses_mlb',
        'Win_Pct_mlb','GB_mlb','WCGB_mlb','Streak_mlb','R_mlb','RA_mlb'
    ])

team_to_abbrev = {
    'Arizona Diamondbacks': 'ARI', 'Atlanta Braves': 'ATL', 'Baltimore Orioles': 'BAL',
    'Boston Red Sox': 'BOS', 'Chicago Cubs': 'CHC', 'Chicago White Sox': 'CWS',
    'Cincinnati Reds': 'CIN', 'Cleveland Guardians': 'CLE', 'Cleveland Indians': 'CLE',
    'Colorado Rockies': 'COL', 'Detroit Tigers': 'DET', 'Houston Astros': 'HOU',
    'Kansas City Royals': 'KC', 'Los Angeles Angels': 'LAA', 'Los Angeles Dodgers': 'LAD',
    'Miami Marlins': 'MIA', 'Milwaukee Brewers': 'MIL', 'Minnesota Twins': 'MIN',
    'New York Mets': 'NYM', 'New York Yankees': 'NYY', 'Oakland Athletics': 'OAK',
    'Philadelphia Phillies': 'PHI', 'Pittsburgh Pirates': 'PIT', 'San Diego Padres': 'SD',
    'San Francisco Giants': 'SF', 'Seattle Mariners': 'SEA', 'St. Louis Cardinals': 'STL',
    'Tampa Bay Rays': 'TB', 'Texas Rangers': 'TEX', 'Toronto Blue Jays': 'TOR',
    'Washington Nationals': 'WSH'
}

all_data = []
years = range(2016, 2026)

for year in years:
    try:
        print(f"📅 Processing {year}...")
        batting = team_batting(year).reset_index()
        pitching = team_pitching(year).reset_index()
        fielding = team_fielding(year).reset_index()

        for dname, d in [('batting',batting), ('pitching',pitching), ('fielding',fielding)]:
            if 'Team' in d.columns:
                d = d[~d['Team'].astype(str).str.contains('Total|League', case=False, na=False)]
            if 'Season' in d.columns and 'Year' not in d.columns:
                d['Year'] = pd.to_numeric(d['Season'], errors='coerce').fillna(year).astype(int)
            elif 'Year' not in d.columns:
                d['Year'] = year
            if dname=='batting': batting=d
            elif dname=='pitching': pitching=d
            else: fielding=d
        has_id = all('teamIDfg' in d.columns for d in (batting, pitching, fielding))

        if has_id:
            key = ['teamIDfg', 'Year']
            merged = batting.merge(pitching, on=key, how='left', suffixes=('_bat', '_pit'))
            f_rename = {c: f'F_{c}' for c in fielding.columns if c not in key + ['Team']}
            fielding = fielding.rename(columns=f_rename)
            merged = merged.merge(fielding, on=key, how='left')
            if 'Team_bat' in merged.columns and 'Team' not in merged.columns:
                merged = merged.rename(columns={'Team_bat': 'Team'})
            merged = merged.drop(columns=['Team_pit'], errors='ignore')
        else:
            for d in (batting, pitching, fielding):
                if 'Team' not in d.columns:
                    raise ValueError("Expected 'Team' column for nickname fallback")
                d['Team_Clean'] = clean_team_string(d['Team'])
                d['Team_Nickname'] = d['Team_Clean'].apply(nickname_from_full)

            key = ['Team_Nickname', 'Year']
            merged = batting.merge(pitching, on=key, how='left', suffixes=('_bat', '_pit'))

            f_rename = {c: f'F_{c}' for c in fielding.columns if c not in key}
            fielding = fielding.rename(columns=f_rename)
            merged = merged.merge(fielding, on=key, how='left')

            if 'Team' not in merged.columns and 'Team_bat' in merged.columns:
                merged = merged.rename(columns={'Team_bat': 'Team'})
            merged = merged.drop(columns=['Team_pit'], errors='ignore')

        if 'Team' in merged.columns:
            nteams = merged['Team'].nunique()
            if year != 2020 and nteams < 28:
                print(f"  ⚠️ Low team count in {year}: {nteams}")

        all_data.append(merged)
        print(f"  ✅ Completed {year} - {len(merged)} teams")

        time.sleep(2)

    except Exception as e:
        print(f"  ❌ Error with {year}: {e}")
        continue

df = pd.concat(all_data, ignore_index=True)

df['Team_Clean'] = clean_team_string(df['Team'])
if 'Team_Nickname' not in df.columns:
    df['Team_Nickname'] = df['Team_Clean'].apply(nickname_from_full)

playoff_results_names = {
    2024: {
        'Dodgers': 5, 'Yankees': 4, 'Guardians': 3, 'Mets': 3,
        'Phillies': 2, 'Padres': 2, 'Tigers': 2, 'Royals': 2,
        'Orioles': 1, 'Astros': 1, 'Brewers': 1, 'Braves': 1
    },
    2023: {
        'Rangers': 5, 'Diamondbacks': 4, 'Phillies': 3, 'Astros': 3,
        'Braves': 2, 'Dodgers': 2, 'Orioles': 2, 'Twins': 2,
        'Rays': 1, 'Blue Jays': 1, 'Marlins': 1, 'Brewers': 1
    },
    2022: {
        'Astros': 5, 'Phillies': 4, 'Padres': 3, 'Yankees': 3,
        'Braves': 2, 'Dodgers': 2, 'Guardians': 2, 'Mariners': 2,
        'Cardinals': 1, 'Mets': 1, 'Rays': 1, 'Blue Jays': 1
    },
    2021: {
        'Braves': 5, 'Astros': 4, 'Dodgers': 3, 'Red Sox': 3,
        'Giants': 2, 'Rays': 2, 'White Sox': 2, 'Brewers': 2,
        'Yankees': 1, 'Cardinals': 1
    },
    2020: {
        'Dodgers': 5, 'Rays': 4, 'Braves': 3, 'Astros': 3,
        'Padres': 2, 'Marlins': 2, 'Yankees': 2, 'Athletics': 2,
        'Cardinals': 1, 'Brewers': 1, 'Twins': 1, 'White Sox': 1,
        'Reds': 1, 'Cubs': 1, 'Indians': 1, 'Blue Jays': 1
    },
    2019: {
        'Nationals': 5, 'Astros': 4, 'Cardinals': 3, 'Yankees': 3,
        'Dodgers': 2, 'Braves': 2, 'Twins': 2, 'Rays': 2,
        'Brewers': 1, 'Athletics': 1
    },
    2018: {
        'Red Sox': 5, 'Dodgers': 4, 'Brewers': 3, 'Astros': 3,
        'Braves': 2, 'Rockies': 2, 'Indians': 2, 'Yankees': 2,
        'Cubs': 1, 'Athletics': 1
    },
    2017: {
        'Astros': 5, 'Dodgers': 4, 'Yankees': 3, 'Cubs': 3,
        'Indians': 2, 'Red Sox': 2, 'Nationals': 2, 'Diamondbacks': 2,
        'Twins': 1, 'Rockies': 1
    },
    2016: {
        'Cubs': 5, 'Indians': 4, 'Dodgers': 3, 'Blue Jays': 3,
        'Rangers': 2, 'Red Sox': 2, 'Giants': 2, 'Nationals': 2,
        'Orioles': 1, 'Mets': 1
    }
}

rows = []
for y, name_outcomes in playoff_results_names.items():
    for full_name, outcome in name_outcomes.items():
        rows.append({'Year': y, 'Team_Clean': full_name, 'playoff_outcome_assigned': outcome})
playoff_df = pd.DataFrame(rows)

df = df.merge(playoff_df, on=['Year', 'Team_Clean'], how='left')
df['playoff_outcome'] = df['playoff_outcome_assigned'].fillna(0).astype(int)
df = df.drop(columns=['playoff_outcome_assigned'])

mlb_runs_all = []
for y in years:
    print(f"  ↪️ Fetching MLB standings (runs) for {y} ...")
    mlb_y = fetch_mlb_standings_runs(y)
    if not mlb_y.empty:
        mlb_runs_all.append(mlb_y)
    time.sleep(0.5)

if mlb_runs_all:
    mlb_df = pd.concat(mlb_runs_all, ignore_index=True)
    df = df.merge(
        mlb_df[['Year','Team_Nickname','R_mlb','RA_mlb','Wins_mlb','Losses_mlb','Win_Pct_mlb']],
        left_on=['Year','Team_Clean'],
        right_on=['Year','Team_Nickname'],
        how='left'
    ).drop(columns=['Team_Nickname_y'], errors='ignore')
    if 'Team_Nickname_x' in df.columns:
        df = df.rename(columns={'Team_Nickname_x': 'Team_Nickname'})
else:
    for c in ['R_mlb','RA_mlb','Wins_mlb','Losses_mlb','Win_Pct_mlb']:
        df[c] = pd.Series(dtype='float64' if 'Pct' in c else 'Int64')

if 'R' not in df.columns and 'R_mlb' in df.columns:
    df['R'] = df['R_mlb'].astype('float')
if 'RA' not in df.columns and 'RA_mlb' in df.columns:
    df['RA'] = df['RA_mlb'].astype('float')

if {'R', 'RA'}.issubset(df.columns):
    df['pythW_pct'] = pyth_win_pct(df['R'], df['RA'], exponent=1.83)
    print("  ✅ Created pythW_pct from runs and runs allowed")

if {'R', 'RA', 'G'}.issubset(df.columns):
    df['run_diff_per_game'] = (df['R'] - df['RA']) / df['G']

for col, default in [('xFIP', np.nan), ('xwOBA', np.nan), ('wRC+', np.nan), ('F_DRS', np.nan), ('F_OAA', np.nan), ('W%', np.nan)]:
    if col not in df.columns:
        df[col] = default

df['Team_Abbrev'] = df['Team_Clean'].map(team_to_abbrev)

modeling_columns = ['Year', 'Team', 'Team_Abbrev']
for col in ['pythW_pct', 'xFIP', 'SIERA', 'xwOBA', 'wRC+', 'F_DRS', 'F_OAA', 'run_diff_per_game',
            'W', 'L', 'R', 'RA', 'playoff_outcome']:
    if col in df.columns:
        modeling_columns.append(col)

final_df = df[modeling_columns].copy()
final_df = final_df.drop(columns=[c for c in ['R_mlb','RA_mlb','Wins_mlb','Losses_mlb','Win_Pct_mlb'] if c in final_df.columns], errors='ignore')
final_df = final_df.drop('Team_Abbrev', axis=1, errors='ignore')

astype_map = {
    'Year': 'int16',
    'W': 'float32', 'L': 'float32', 'R': 'float32', 'RA': 'float32',
    'pythW_pct': 'float32',
    'run_diff_per_game': 'float32',
    'playoff_outcome': 'int8'
}
final_df = final_df.astype({k:v for k,v in astype_map.items() if k in final_df.columns}, errors='ignore')

filename = 'mlb_playoff_data_2016_2025.csv'
final_df.to_csv(filename, index=False)
print(f"\n Data saved to {filename}")

import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.miscmodels.ordinal_model import OrderedModel

df = pd.read_csv("mlb_playoff_data_2016_2025.csv")

train = df[df["Year"] <= 2024].copy()
test = df[df["Year"] == 2025].copy()

playoff_teams_2025 = ["Dodgers","Yankees","Blue Jays","Tigers","Mariners","Red Sox","Astros","Brewers","Phillies","Cubs","Padres","Mets"]
test[test["Team"].isin(playoff_teams_2025)].copy()

candidate_feats = ["pythW_pct","run_diff_per_game","wRC+","xwOBA","xFIP","SIERA","F_DRS","F_OAA","W","L","R","RA"]
feat_cols = [c for c in candidate_feats if c in df.columns]

X_tr = train[feat_cols].fillna(0.0)
y_tr = train["playoff_outcome"].astype(int)

mod = OrderedModel(y_tr, X_tr, distr="logit")
res = mod.fit(method="bfgs", maxiter=200, disp=False)

X_te = test[feat_cols].fillna(0.0)
probs = res.predict(X_te, which="prob")

K = probs.shape[1]
prob_cols = [f"p{i}" for i in range(K)]
test[prob_cols] = probs * 100
test["pred_stage"] = probs.values.argmax(axis=1)

out_cols = ["Year","Team"] + prob_cols + ["pred_stage"]
print(test[out_cols].sort_values("p5", ascending=False))
