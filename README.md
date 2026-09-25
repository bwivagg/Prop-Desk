# Prop Desk

A player prop projection and line-setting tool, built from the operator's side of the counter. For every player on the NFL slate it projects outcomes across 8 markets, posts a balanced line, and flags props that need human review before they go live.

**Live app:** _add your Streamlit link here_

## What it does

- **Line board.** Projections and posted lines for Pass Yards, Pass TDs, Pass Attempts, Rush Yards, Receptions, Receiving Yards, Rush + Rec Yards, and Fantasy Score. Lines can be overridden by hand and exported.
- **Risk queue.** Flags injuries, vacated volume when a teammate is out, role changes, new teams, blowout spreads, weather, and high volatility.
- **Weather desk.** Enter forecast wind for outdoor games and projections update.
- **Player view.** Breaks each line into baseline, matchup, game environment, and weather adjustments, next to the player's game log.
- **Model accuracy.** An out-of-sample backtest on the 2025 season.

## Method

1. Recency-weighted baseline blended with last season. The blend adjusts when a player's role or team changes.
2. Opponent adjustment by position, shrunk toward league average and capped at ±15%.
3. Vegas implied team total and spread for game environment and game script.
4. Wind and cold adjustments for outdoor games.
5. Lines calibrated to the median outcome per market, since yardage is right-skewed, then rounded to .5.

## Results (2025 season, weeks 4–18, out-of-sample)

- **8,430 props graded.**
- **Over rate of 47.7%.** A line posted at the raw average let overs hit only about 42% on yardage markets. Median calibration was tuned on 2024 and tested on 2025.
- **1.8% lower average error than a season-to-date average**, with the largest gains on pass attempts (4.9%), receptions, and receiving yards (3.3%).

## Roadmap

- Snap share and route participation inputs, projecting volume × efficiency
- NBA module, with the same engine and new markets (PTS, REB, AST, PRA)
- MLB and NHL modules
- Live in-game line updates

## Data

[nflverse](https://github.com/nflverse): weekly player stats, schedules with closing lines, and injury reports.

## Stack

Python, pandas, Streamlit, Altair
