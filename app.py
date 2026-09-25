"""Prop Desk - player prop projection and line-setting board."""

import altair as alt
import pandas as pd
import streamlit as st

import nfl_model as nfl

SEASON = 2026
SPORTS = {"NFL": nfl}  # new sports plug in here with the same interface

st.set_page_config(page_title="Prop Desk", page_icon="🏈", layout="wide")

st.markdown("""
<style>
  .block-container {padding-top: 2rem; max-width: 1400px;}
  h1 {font-weight: 800; letter-spacing: -0.02em; margin-bottom: 0;}
  .sub {color: #5B6475; font-size: 1.02rem; margin-top: .25rem; margin-bottom: 1.25rem;}
  div[data-testid="stMetricValue"] {font-variant-numeric: tabular-nums;}
</style>
""", unsafe_allow_html=True)


# ------------------------------------------------------------------ data
@st.cache_data(ttl=6 * 3600, show_spinner="Pulling latest NFL data…")
def load():
    return nfl.load_data((SEASON - 2, SEASON - 1, SEASON))


@st.cache_data(ttl=3600, show_spinner="Building projections…")
def projections(week, wind_overrides):
    stats, games, inj = load()
    return nfl.project_slate(stats, games, inj, SEASON, week, dict(wind_overrides))


@st.cache_data(show_spinner="Grading every 2025 projection against what happened…")
def run_backtest():
    stats, games, _ = load()
    return nfl.backtest(stats, games, SEASON - 1, range(4, 19))


try:
    stats, games, inj = load()
except Exception as e:
    st.error(f"Couldn't reach the nflverse data feed ({e}). Refresh in a minute; the feed updates overnight.")
    st.stop()

# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.header("Slate")
    sport = st.selectbox("Sport", list(SPORTS))
    live_week = nfl.current_week(games, SEASON)
    week = st.number_input("Week", 1, 18, live_week)

    slate = nfl.slate_games(games, SEASON, week).drop_duplicates("game_id")
    st.header("Weather desk")
    st.caption("Forecast wind isn't in the feed until kickoff. Enter it for outdoor games; "
               "15+ mph cuts passing and receiving projections.")
    overrides = []
    outdoor = slate[slate["roof"].isin(["outdoors", "open"])]
    for _, g in outdoor.iterrows():
        default = int(g["wind"]) if pd.notna(g["wind"]) else 0
        w = st.slider(g["matchup"], 0, 35, default, key=f"wind_{g['game_id']}")
        if w != default or pd.isna(g["wind"]):
            overrides.append((g["game_id"], w))

board = projections(week, tuple(overrides))
if board.empty:
    st.warning(f"No games found for week {week}. Pick another week.")
    st.stop()

# ------------------------------------------------------------------ header
st.title("Prop Desk")
st.markdown(f"<div class='sub'>{SEASON} NFL week {week}: {slate.shape[0]} games, "
            f"{len(board):,} projected props. Lines update when the data feed refreshes.</div>",
            unsafe_allow_html=True)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Ready to post", int((board["status"] == "Post").sum()))
c2.metric("Needs review", int((board["status"] == "Review").sum()))
c3.metric("Pull (injury)", int((board["status"] == "Pull").sum()))
c4.metric("High confidence", int((board["confidence"] == "High").sum()))

tab_board, tab_risk, tab_player, tab_acc, tab_how = st.tabs(
    ["Line board", "Risk queue", "Player view", "Model accuracy", "How it works"])

# ------------------------------------------------------------------ line board
with tab_board:
    f1, f2, f3, f4 = st.columns([2, 2, 2, 3])
    mkts = f1.multiselect("Market", list(nfl.MARKETS), default=["Pass Yards", "Receiving Yards", "Rush Yards"])
    games_pick = f2.multiselect("Game", sorted(board["matchup"].unique()))
    status_pick = f3.multiselect("Status", ["Post", "Review", "Pull"], default=["Post", "Review"])
    search = f4.text_input("Player search")

    view = board.copy()
    if mkts:
        view = view[view["market"].isin(mkts)]
    if games_pick:
        view = view[view["matchup"].isin(games_pick)]
    if status_pick:
        view = view[view["status"].isin(status_pick)]
    if search:
        view = view[view["player"].str.contains(search, case=False)]
    view = view.sort_values(["market", "projection"], ascending=[True, False])
    view["override"] = None

    st.caption("Projection is the model's expected outcome. The line is calibrated from it so overs and "
               "unders split about evenly (tuned on 2024 results). Type in the Override column to move a line by hand.")
    edited = st.data_editor(
        view[["player", "pos", "team", "opp", "market", "projection", "line", "override",
              "confidence", "status", "risk"]],
        hide_index=True, width="stretch", height=560,
        disabled=["player", "pos", "team", "opp", "market", "projection", "line", "confidence", "status", "risk"],
        column_config={
            "player": "Player", "pos": "Pos", "team": "Team", "opp": "Opp", "market": "Market",
            "confidence": "Confidence", "status": "Status",
            "projection": st.column_config.NumberColumn("Projection", format="%.1f"),
            "line": st.column_config.NumberColumn("Line", format="%.1f"),
            "override": st.column_config.NumberColumn("Override", step=0.5, format="%.1f"),
            "risk": st.column_config.TextColumn("Risk flags", width="large"),
        },
    )
    final = edited.assign(final_line=edited["override"].fillna(edited["line"]))
    moved = final["override"].notna().sum()
    st.download_button(f"Export board ({len(final)} lines, {moved} overridden)",
                       final.to_csv(index=False), f"prop_board_week{week}.csv", "text/csv")

# ------------------------------------------------------------------ risk queue
with tab_risk:
    st.subheader("What needs a look before lines go live")
    flagged = board[board["risk"] != ""]
    kinds = (flagged["risk"].str.split(", ").explode()
             .str.replace(r"[:(].*", "", regex=True).str.replace(r"\d+.*", "", regex=True).str.strip())
    counts = kinds.value_counts().rename_axis("flag").reset_index(name="props")
    left, right = st.columns([1, 2])
    left.dataframe(counts, hide_index=True, width="stretch")
    right.dataframe(
        flagged.sort_values(["status", "matchup"])[["status", "player", "team", "market", "line", "risk"]],
        hide_index=True, width="stretch", height=420)
    with st.expander("What each flag means"):
        st.markdown("""
- **Out / Doubtful / Questionable**: on the official injury report. Out and Doubtful props are pulled.
- **Vacated volume**: a teammate who held 15%+ of targets or 25%+ of carries is out. Usage will shift, so the line is soft.
- **Role change**: volume this season is 40%+ different from last season. Less weight goes on last year's numbers.
- **New team**: new scheme and new teammates. Last season is weighted less.
- **Blowout risk**: spread of 9.5+. Starters can sit late and game script gets extreme.
- **Wind / Cold**: outdoor game conditions that suppress passing.
- **High volatility**: this player's game-to-game swings are big relative to his average.
- **Small sample**: rookie or new starter with fewer than 3 games.
""")

# ------------------------------------------------------------------ player view
with tab_player:
    p_name = st.selectbox("Player", sorted(board["player"].unique()))
    p_rows = board[board["player"] == p_name]
    p_mkt = st.selectbox("Market", p_rows["market"].tolist())
    r = p_rows[p_rows["market"] == p_mkt].iloc[0]
    stat = nfl.MARKETS[p_mkt]["stat"]

    a, b, c, d, e = st.columns(5)
    a.metric("Baseline", f"{r['base']:.1f}")
    b.metric("Matchup", f"{(r['opp_adj'] - 1) * 100:+.0f}%", help=f"What {r['opp']} allows to this position vs league average")
    c.metric("Game environment", f"{(r['env_adj'] - 1) * 100:+.0f}%", help="Vegas implied team total and spread")
    d.metric("Weather", f"{(r['weather_adj'] - 1) * 100:+.0f}%")
    e.metric("Posted line", f"{r['line']:.1f}", help=f"Projection {r['projection']:.1f}, calibrated to a balanced line")
    if r["risk"]:
        st.warning(f"Risk flags: {r['risk']}")

    log = stats[(stats["player_id"] == r["player_id"]) & (stats["season"] >= SEASON - 1)
                & ~((stats["season"] == SEASON) & (stats["week"] >= week))].sort_values("order")
    log = log.assign(game="'" + (log["season"] % 100).astype(str) + " W" + log["week"].astype(str) + " " + log["opponent_team"])
    bars = alt.Chart(log).mark_bar(color="#1F4FD1").encode(
        x=alt.X("game:N", sort=None, title=None, axis=alt.Axis(labelAngle=-45)), y=alt.Y(f"{stat}:Q", title=p_mkt),
        tooltip=["game", stat])
    rule = alt.Chart(pd.DataFrame({"y": [r["line"]]})).mark_rule(color="#B7791F", strokeWidth=2, strokeDash=[6, 4]).encode(y="y:Q")
    st.altair_chart(bars + rule, width="stretch")
    over = (log[stat] > r["line"]).mean() if len(log) else float("nan")
    st.caption(f"Dashed line = this week's posted line. {p_name} went over it in {over:.0%} of the games shown.")

# ------------------------------------------------------------------ accuracy
with tab_acc:
    st.subheader("Out-of-sample test: 2025 season, weeks 4–18")
    st.markdown("Every week is projected using only data available before kickoff, then graded against "
                "what actually happened. The baseline to beat is the player's season-to-date average. "
                "Line shading was calibrated on 2024, so 2025 is a clean test.")
    if st.button("Run backtest"):
        bt, summary = run_backtest()
        tot = summary["props"].sum()
        k1, k2, k3 = st.columns(3)
        k1.metric("Props graded", f"{tot:,}")
        k2.metric("Overs hit", f"{bt['over'].mean():.1%}", help="A balanced board lands near 50%")
        imp = (summary["mae_improvement"] * summary["props"]).sum() / tot
        k3.metric("Error vs season average", f"{-imp:+.1%}", help="Mean absolute error vs the naive baseline")
        show = summary.reset_index().rename(columns={
            "market": "Market", "props": "Props", "model_mae": "Model error",
            "naive_mae": "Season-avg error", "over_rate": "Over rate", "mae_improvement": "Improvement"})
        st.dataframe(show.style.format({"Model error": "{:.2f}", "Season-avg error": "{:.2f}",
                                        "Over rate": "{:.1%}", "Improvement": "{:+.1%}"}),
                     hide_index=True, width="stretch")
        st.caption("Before median shading, overs hit 42–43% on yardage markets. That's the gap a sharp "
                   "bettor would exploit by hammering unders.")

# ------------------------------------------------------------------ method
with tab_how:
    st.markdown("""
#### How a line gets built

1. **Baseline.** Recency-weighted average of this season's games (each older game counts 15% less),
   blended with last season as a 4-game prior. If a player's role or team changed, last season counts less.
2. **Matchup.** What the opponent allows to that position compared with league average, shrunk
   halfway toward average and capped at ±15% so small samples don't swing lines.
3. **Game environment.** Vegas implied team total scales volume and efficiency. The spread adjusts
   for game script: favorites run more, underdogs throw more.
4. **Weather.** Outdoor wind of 15+ mph (−7%) or 20+ mph (−13%) and temperatures at or below 25°F cut passing.
5. **Line.** Yardage outcomes are skewed by occasional big games, so a line posted at the average gets
   bet under too often. Each market's line is calibrated to the median outcome (tuned on the 2024 season)
   and rounded to a .5 so nothing pushes.
6. **Risk review.** Injuries, vacated volume, role changes, blowout spreads and volatility get flagged
   for a human to check before the line is posted.

**Data:** nflverse weekly player stats, schedules with closing lines, and official injury reports.
Refreshes every 6 hours.
""")
