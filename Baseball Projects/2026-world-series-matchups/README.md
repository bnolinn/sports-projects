# 2026 World Series Preview: Rays vs. Brewers

Batter-vs-pitcher matchup analysis for the 2026 World Series using career Statcast data.

## Question
Which hitters have historically handled the opposing team's pitching staff, and where are the lopsided matchups?

## Data
- Statcast pitch-level data pulled with [pybaseball](https://github.com/jldbc/pybaseball)
- Career head-to-head plate appearances between each team's hitters and the opposing pitchers

## Method
- Used **xwOBA** rather than wOBA. It's based on quality of contact (exit velocity and launch angle), so it filters out luck from defense and ballparks. That matters in small head-to-head samples.
- Built a hitter × pitcher matrix for each team and plotted it as a heatmap.

## Results

### Rays hitters vs. Brewers pitchers
![Rays hitters vs Brewers pitchers](images/IMG_1689.png)

### Brewers hitters vs. Rays pitchers
![Brewers hitters vs Rays pitchers](images/IMG_1688.png)

## Limitations
- Career head-to-head samples are small, so many matchups have under 10 plate appearances.
- Doesn't account for changes in a player's skill over time.

## Tools
Python · pandas · pybaseball · matplotlib/seaborn · Google Colab