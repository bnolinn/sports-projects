# Sports Analytics Projects — Brady Nolin

Prediction models, matchup analysis, and data visualization across MLB, NFL, and NHL, built in Python with public data (Statcast, MLB Stats API, NHL data).

## Featured projects

| Project | Sport | What it does | Tools |
|---|---|---|---|
| [2026 World Series matchup model](Baseball%20Projects/2026-world-series-matchups/) | MLB | Projects every Rays–Brewers hitter-vs-pitcher matchup by finding pitches that look like each pitcher's arsenal (nearest-neighbor matching on velocity, spin, and movement) and measuring the hitter's xwOBA against them. | pandas, NumPy, matplotlib, pybaseball |
| [Matchup Lab dashboard](Baseball%20Projects/Matchup%20Dashboard/) | MLB | Interactive dashboard for any hitter-vs-pitcher or team-vs-team matchup. It weights every pitch a hitter has seen by how closely it matches each pitch in the pitcher's arsenal (velocity, movement, spin, release, extension), then projects run value, whiff, chase, and xwOBA on contact, plus a location fit. | Streamlit, Plotly, pandas, NumPy, pybaseball |
| [2025 MLB playoff model](Baseball%20Projects/Prediction%20Models/2025%20Playoff%20Model/) | MLB | Ordered logistic regression trained on 2016–2024 playoff teams to estimate how far each 2025 playoff team goes, plus a series simulator. | pandas, statsmodels, pybaseball, MLB Stats API |
| [2026 NFL Big Data Bowl](Football%20Projects/2026%20NFL%20Big%20Data%20Bowl/) | NFL | Situational decision-making analysis: two-minute drill, red zone, fourth downs, and receiver separation. | pandas, scikit-learn, Jupyter |
| [NHL late-lead research](Hockey%20Projects/late-lead-extensions/) | NHL | How often a 2-goal lead with under 6 minutes left turns into a 3+ goal win once the trailing team pulls its goalie. *In progress.* | Python |

## Repository structure

```
sports-projects/
├── Baseball Projects/
│   ├── 2026-world-series-matchups/     
│   ├── Matchup Dashboard/
│   └── Prediction Models/
│       └── 2025 Playoff Model/       
├── Football Projects/
│   └── 2026 NFL Big Data Bowl/
├── Hockey Projects/
│   └── late-lead-extensions/
└── Basketball Projects/
```

## Running the code

Each project folder has its own README with method and results.

```
pip install pybaseball pandas numpy matplotlib statsmodels
```

## Data sources

- [Baseball Savant / Statcast](https://baseballsavant.mlb.com) via pybaseball
- [FanGraphs](https://www.fangraphs.com) team stats via pybaseball
- [MLB Stats API](https://statsapi.mlb.com)
- [NFL Big Data Bowl](https://operations.nfl.com/gameday/analytics/big-data-bowl/) tracking data

---

Questions or feedback? Reach out on [GitHub](https://github.com/bnolinn).
