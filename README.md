# Sports Analytics Projects — Brady Nolin

Prediction models, matchup analysis, and data visualization across MLB, NFL, and NHL, built in Python with public data (Statcast, MLB Stats API, NHL data).

## Featured projects

| Project | Sport | What it does | Tools |
|---|---|---|---|
| [2026 World Series matchup model](Baseball%20Projects/2026-world-series-matchups/) | MLB | Projects every Rays–Brewers hitter-vs-pitcher matchup by finding pitches that look like each pitcher's arsenal (nearest-neighbor matching on velocity, spin, and movement) and measuring the hitter's xwOBA against them. | pandas, NumPy, matplotlib, pybaseball |
| [2025 MLB playoff model](Baseball%20Projects/Prediction%20Models/2025%20Playoff%20Model/) | MLB | Ordered logistic regression trained on 2016–2024 playoff teams to estimate how far each 2025 playoff team goes, plus a series simulator. | pandas, statsmodels, pybaseball, MLB Stats API |
| [2026 NFL Big Data Bowl](Football%20Projects/2026%20NFL%20Big%20Data%20Bowl/) | NFL | Situational decision-making analysis: two-minute drill, red zone, fourth downs, and receiver separation. | pandas, scikit-learn, Jupyter |
| [NHL late-lead research](Hockey%20Projects/late-lead-extensions/) | NHL | How often a 2-goal lead with under 6 minutes left turns into a 3+ goal win once the trailing team pulls its goalie. *In progress.* | Python |

## Repository structure

```
sports-projects/
├── Baseball Projects/
│   ├── 2026-world-series-matchups/     xwOBA matchup heatmaps (script + Colab notebook)
│   └── Prediction Models/
│       └── 2025 Playoff Model/         data pull, model, series simulator
├── Football Projects/
│   └── 2026 NFL Big Data Bowl/         linked repository (git submodule)
├── Hockey Projects/
│   └── late-lead-extensions/
└── Basketball Projects/                coming soon
```

## Running the code

Each project folder has its own README with method and results. The baseball projects use [pybaseball](https://github.com/jldbc/pybaseball):

```
pip install pybaseball pandas numpy matplotlib statsmodels
```

The World Series notebook is set up to run in Google Colab.

## Data sources

- [Baseball Savant / Statcast](https://baseballsavant.mlb.com) via pybaseball
- [FanGraphs](https://www.fangraphs.com) team stats via pybaseball
- [MLB Stats API](https://statsapi.mlb.com)
- [NFL Big Data Bowl](https://operations.nfl.com/gameday/analytics/big-data-bowl/) tracking data

---

Questions or feedback? Reach out on [GitHub](https://github.com/bnolinn).
