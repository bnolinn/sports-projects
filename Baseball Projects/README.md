# Baseball Projects

- **[2026 World Series matchup model](2026-world-series-matchups/)**: projected xwOBA for every Rays–Brewers hitter-vs-pitcher matchup, built from pitch-shape look-alikes instead of tiny head-to-head samples.
- **[Matchup Lab dashboard](Matchup%20Dashboard/)**: Streamlit app for any hitter-vs-pitcher or team-vs-team matchup. It uses a pitch-shape similarity model on count-neutral run value, whiff, chase, and xwOBA on contact, plus location fit.
- **[2025 playoff model](Prediction%20Models/2025%20Playoff%20Model/)**: ordered logistic regression on team metrics (Pythagorean win %, wRC+, xwOBA, xFIP, SIERA, DRS, OAA) trained on 2016–2024 playoff teams, with a series simulator.
- **[Next-season wOBA predictors](wRC%2B%20Predictors/)**: ranks ~60 offensive stats from Baseball Savant by how well a hitter's value this season predicts his wOBA next season (Pearson r, 2015–2025 qualified hitters), to show which stats reflect repeatable skill. Charts the top 10. Run `python wrc_predictors.py`.
