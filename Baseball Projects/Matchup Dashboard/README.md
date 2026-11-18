# MLB Matchup Lab

```bash
pip install pybaseball streamlit plotly pandas numpy pyarrow matplotlib
python build_data.py --year 2026
streamlit run app.py
```

## Views

| View | What it answers |
|---|---|
| **Team vs Team** | Heatmap of every hitter vs every pitcher in both directions, best spots for each side, lineup/staff averages |
| **Matchup Breakdown** | One batter vs one pitcher: edge verdict, advantages for each side, per-pitch model, movement map, zone damage vs location, arsenal, head-to-head |
| **Pitcher Game Plan** | A pitcher vs an opposing lineup: per-hitter edge, pitch to attack with, pitch to avoid, put-away pitch |
| **Hitter Scouting Report** | A hitter vs an opposing staff: per-pitcher edge, pitch to hunt, pitch to lay off |

## Model

Everything is in `matchup_engine.py`.

1. **Count-neutral run value**: Statcast `delta_run_exp` minus the league average for that ball-strike count, shown per 100 pitches from the hitter's perspective (+ = hitter edge).
2. **Stuff baseline**: each pitch in the arsenal is a vector of velocity, induced vertical break, arm-side break, spin, release height/side, and extension. Every league pitch from a same-handed pitcher to a same-sided hitter is weighted by a Gaussian kernel on that vector. The result is what an average hitter does against that exact shape.
3. **Pitcher estimate**: his actual results with that pitch against that side, shrunk toward the stuff baseline.
4. **Hitter estimate**: the same kernel applied to every pitch the hitter has seen, so a hitter who does damage against 97 mph high-ride fastballs gets credit against any pitch with that shape, whatever its label. Also shrunk toward league.
5. **Combine**: pitcher and hitter effects add (on log-odds for whiff/chase rates), then get weighted by the pitcher's usage against that batter side.
6. **Location fit**: the pitcher's location density multiplied by the hitter's smoothed zone damage map.

Kernel bandwidths and shrinkage constants are at the top of `matchup_engine.py`.
