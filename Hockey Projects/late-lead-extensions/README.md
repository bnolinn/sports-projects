# NHL Late-Lead Extensions

*Status: in progress*

## Question
When an NHL team leads by exactly 2 goals with under 6 minutes left in the third period, how often does the game end as a 3+ goal win?

## Why it matters
Trailing teams pull their goalie late, which raises their chance of scoring but also opens up empty-net goals for the leader. This project measures how often that trade-off ends with the leader extending the lead, which matters for puck-line (−1.5 / −2.5) and live betting markets.

## Plan
1. Pull play-by-play for recent NHL seasons and find every game state of "leading by 2, under 6:00 left in the 3rd."
2. Record the final margin of each game, and whether and when the goalie was pulled.
3. Break results down by time remaining, home/away, and team strength.
4. Build a real-time alert that posts when a game hits that state.
