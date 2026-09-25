"""
NFL projection engine for Prop Desk.

Pipeline for one slate:
  1. Baseline   - recency-weighted per-game average, blended with last season
  2. Matchup    - opponent's allowed production to that position vs league average
  3. Game env   - Vegas implied team total + spread (game script)
  4. Weather    - wind / cold for outdoor games
  5. Line       - projection rounded to a posted x.5 line
  6. Risk       - flags a trader should review before the line goes live
"""

import math
import numpy as np
import pandas as pd

BASE = "https://github.com/nflverse/nflverse-data/releases/download"
GAMES_URL = "https://github.com/nflverse/nfldata/raw/master/data/games.csv"

# ---------------------------------------------------------------- markets
# volume = the usage stat used for eligibility and role-change detection
MARKETS = {
    "Pass Yards":       dict(stat="passing_yards",  pos=["QB"],             volume="attempts", min_vol=15, kind="pass"),
    "Pass TDs":         dict(stat="passing_tds",    pos=["QB"],             volume="attempts", min_vol=15, kind="pass"),
    "Pass Attempts":    dict(stat="attempts",       pos=["QB"],             volume="attempts", min_vol=15, kind="volume_pass"),
    "Rush Yards":       dict(stat="rushing_yards",  pos=["QB", "RB"],       volume="carries",  min_vol=5,  kind="rush"),
    "Receptions":       dict(stat="receptions",     pos=["RB", "WR", "TE"], volume="targets",  min_vol=3,  kind="volume_rec"),
    "Receiving Yards":  dict(stat="receiving_yards", pos=["RB", "WR", "TE"], volume="targets", min_vol=3,  kind="rec"),
    "Rush + Rec Yards": dict(stat="rush_rec_yards", pos=["RB"],             volume="touches",  min_vol=8,  kind="rush"),
    "Fantasy Score":    dict(stat="pp_fantasy",     pos=["QB", "RB", "WR", "TE"], volume="touches", min_vol=6, kind="fantasy"),
}

# how much each adjustment moves each kind of market
ENV_EXP = {"pass": 0.6, "volume_pass": 0.3, "rush": 0.5, "volume_rec": 0.3, "rec": 0.6, "fantasy": 0.7}
SCRIPT = {"pass": -0.005, "volume_pass": -0.008, "rush": 0.008, "volume_rec": -0.004, "rec": -0.005, "fantasy": 0.0}
WEATHER_HIT = {"pass", "rec", "fantasy"}

# Most prop outcomes are right-skewed (a few blowup games pull the average up),
# so a line posted at the average gets bet under too often. Lines are shaded to the
# median. Factors = median(actual / projection), calibrated on the 2024 season and
# tested out-of-sample on 2025.
LINE_SHADE = {
    "Pass Yards": 1.035, "Pass TDs": 1.0, "Pass Attempts": 1.02, "Rush Yards": 0.85,
    "Receptions": 0.93, "Receiving Yards": 0.886, "Rush + Rec Yards": 0.926, "Fantasy Score": 0.919,
}

PRIOR_GAMES = 4.0      # last season counts as this many current-season games
RECENCY_DECAY = 0.85   # each older game counts 15% less
OPP_SHRINK = 0.5       # trust half of the observed defensive split


# ---------------------------------------------------------------- data
def load_data(seasons=(2024, 2025, 2026)):
    frames = []
    for s in seasons:
        df = pd.read_csv(f"{BASE}/stats_player/stats_player_week_{s}.csv", low_memory=False)
        frames.append(df)
    stats = pd.concat(frames, ignore_index=True)
    stats = stats[stats["season_type"] == "REG"].copy()
    stats = add_derived(stats)

    games = pd.read_csv(GAMES_URL)
    games = games[(games["season"].isin(seasons)) & (games["game_type"] == "REG")].copy()

    try:
        inj = pd.read_csv(f"{BASE}/injuries/injuries_{max(seasons)}.csv")
    except Exception:
        inj = pd.DataFrame(columns=["season", "week", "gsis_id", "report_status", "report_primary_injury"])
    return stats, games, inj


def add_derived(df):
    for c in ["passing_yards", "passing_tds", "passing_interceptions", "rushing_yards", "rushing_tds",
              "receptions", "receiving_yards", "receiving_tds", "carries", "targets", "attempts",
              "rushing_fumbles_lost", "receiving_fumbles_lost", "sack_fumbles_lost"]:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)
    df["rush_rec_yards"] = df["rushing_yards"] + df["receiving_yards"]
    df["touches"] = df["carries"] + df["targets"] + df["attempts"] * 0.3
    fumbles = df["rushing_fumbles_lost"] + df["receiving_fumbles_lost"] + df["sack_fumbles_lost"]
    # PrizePicks-style NFL fantasy scoring
    df["pp_fantasy"] = (df["passing_yards"] * 0.04 + df["passing_tds"] * 4 - df["passing_interceptions"]
                        + df["rushing_yards"] * 0.1 + df["rushing_tds"] * 6
                        + df["receptions"] + df["receiving_yards"] * 0.1 + df["receiving_tds"] * 6
                        - fumbles)
    df["pos_group"] = df["position"].where(df["position"].isin(["QB", "RB", "WR", "TE"]), "OTHER")
    df["order"] = df["season"] * 100 + df["week"]
    return df


def current_week(games, season):
    g = games[games["season"] == season]
    open_weeks = g[g["home_score"].isna()]["week"]
    return int(open_weeks.min()) if len(open_weeks) else int(g["week"].max())


# ---------------------------------------------------------------- slate context
def slate_games(games, season, week):
    g = games[(games["season"] == season) & (games["week"] == week)].copy()
    g["home_implied"] = (g["total_line"] + g["spread_line"]) / 2
    g["away_implied"] = (g["total_line"] - g["spread_line"]) / 2
    rows = []
    for _, r in g.iterrows():
        common = dict(game_id=r["game_id"], matchup=f"{r['away_team']} @ {r['home_team']}",
                      roof=r["roof"], temp=r["temp"], wind=r["wind"], total=r["total_line"],
                      gameday=r["gameday"], gametime=r["gametime"])
        rows.append(dict(team=r["home_team"], opp=r["away_team"], implied=r["home_implied"],
                         margin=r["spread_line"], **common))
        rows.append(dict(team=r["away_team"], opp=r["home_team"], implied=r["away_implied"],
                         margin=-r["spread_line"], **common))
    return pd.DataFrame(rows)


def defense_factors(hist, season, stat, pos_list):
    """Per-defense multiplier: what they allow to these positions vs league average."""
    h = hist[hist["pos_group"].isin(pos_list)]
    h = h.assign(w=np.where(h["season"] == season, 2.0, 1.0))
    per_game = h.groupby(["opponent_team", "game_id"]).agg(v=(stat, "sum"), w=("w", "first")).reset_index()
    per_game["wv"] = per_game["v"] * per_game["w"]
    d = per_game.groupby("opponent_team").agg(wv=("wv", "sum"), w=("w", "sum"))
    d["allowed"] = d["wv"] / d["w"]
    league = d["allowed"].mean()
    raw = d["allowed"] / league if league > 0 else 1.0
    return (1 + (raw - 1) * OPP_SHRINK).clip(0.85, 1.15)


# ---------------------------------------------------------------- projections
def weighted_mean_std(values):
    n = len(values)
    if n == 0:
        return np.nan, np.nan, 0.0
    w = RECENCY_DECAY ** np.arange(n)[::-1]  # oldest first, newest gets weight 1
    m = np.average(values, weights=w)
    sd = math.sqrt(np.average((values - m) ** 2, weights=w)) if n > 1 else np.nan
    return m, sd, w.sum()


def project_slate(stats, games, inj, season, week, weather_override=None):
    """Return one row per player x market with projection, line, confidence and flags."""
    hist = stats[(stats["season"] < season) | ((stats["season"] == season) & (stats["week"] < week))]
    hist = hist[hist["season"] >= season - 1]
    slate = slate_games(games, season, week)
    if slate.empty:
        return pd.DataFrame()
    team_ctx = slate.set_index("team")
    avg_implied = slate["implied"].mean()
    weather_override = weather_override or {}

    cur = hist[hist["season"] == season]
    prev = hist[hist["season"] == season - 1]

    # most recent team per player this season
    latest = cur.sort_values("order").groupby("player_id").tail(1).set_index("player_id")
    last_prev_team = prev.sort_values("order").groupby("player_id")["team"].last()

    # injury report for this week
    wk_inj = inj[(inj["season"] == season) & (inj["week"] == week)] if len(inj) else inj
    status = wk_inj.set_index("gsis_id")["report_status"].to_dict() if len(wk_inj) else {}
    injury = wk_inj.set_index("gsis_id")["report_primary_injury"].to_dict() if len(wk_inj) else {}

    # vacated volume: an OUT player who held a real share of team targets/carries
    vacated = {}
    if len(cur):
        tot = cur.groupby("team")[["targets", "carries"]].sum()
        ply = cur.groupby(["player_id", "team"])[["targets", "carries"]].sum().reset_index()
        for _, r in ply.iterrows():
            if status.get(r["player_id"]) in ("Out", "Doubtful") and r["team"] in tot.index:
                t_share = r["targets"] / max(tot.loc[r["team"], "targets"], 1)
                c_share = r["carries"] / max(tot.loc[r["team"], "carries"], 1)
                if t_share >= 0.15 or c_share >= 0.25:
                    name = latest.loc[r["player_id"], "player_display_name"] if r["player_id"] in latest.index else "starter"
                    vacated.setdefault(r["team"], []).append(name)

    cur_by = {k: v.sort_values("order") for k, v in cur.groupby("player_id")}
    prev_by = {k: v.sort_values("order") for k, v in prev.groupby("player_id")}
    empty = cur.iloc[0:0]

    out = []
    for market, cfg in MARKETS.items():
        stat, vol, kind = cfg["stat"], cfg["volume"], cfg["kind"]
        dfac = defense_factors(hist, season, stat, cfg["pos"])

        pool = latest[latest["pos_group"].isin(cfg["pos"]) & latest["team"].isin(team_ctx.index)]
        for pid, p in pool.iterrows():
            g_cur = cur_by.get(pid, empty)
            g_prev = prev_by.get(pid, empty)
            recent_vol = g_cur[vol].tail(3).mean()
            if recent_vol < cfg["min_vol"]:
                continue
            if market == "Rush Yards" and p["pos_group"] == "QB" and g_cur["carries"].tail(3).mean() < 3:
                continue

            m_cur, sd_cur, w_cur = weighted_mean_std(g_cur[stat].to_numpy(float))
            m_prev = g_prev[stat].mean() if len(g_prev) else np.nan
            flags = []

            # role change: this season's volume is very different from last season's
            prior_k = PRIOR_GAMES
            if len(g_prev) >= 4:
                ratio = g_cur[vol].mean() / max(g_prev[vol].mean(), 0.1)
                if ratio >= 1.4 or ratio <= 0.6:
                    flags.append("Role change")
                    prior_k = 1.5
            if len(g_prev) and pid in last_prev_team.index and last_prev_team[pid] != p["team"]:
                flags.append("New team")
                prior_k = min(prior_k, 2.0)

            if np.isnan(m_prev):
                base = m_cur
                if len(g_cur) < 3:
                    flags.append("Small sample")
            else:
                base = (w_cur * m_cur + prior_k * m_prev) / (w_cur + prior_k)

            ctx = team_ctx.loc[p["team"]]
            opp_f = float(dfac.get(ctx["opp"], 1.0))
            env_f = float(np.clip((ctx["implied"] / avg_implied) ** ENV_EXP[kind], 0.85, 1.15)) \
                if pd.notna(ctx["implied"]) else 1.0
            script_f = float(np.clip(1 + SCRIPT[kind] * ctx["margin"], 0.9, 1.1)) if pd.notna(ctx["margin"]) else 1.0

            wind = weather_override.get(ctx["game_id"], ctx["wind"])
            wx_f = 1.0
            outdoors = ctx["roof"] in ("outdoors", "open")
            if outdoors and kind in WEATHER_HIT and pd.notna(wind):
                if wind >= 20:
                    wx_f = 0.87
                elif wind >= 15:
                    wx_f = 0.93
                if wx_f < 1:
                    flags.append(f"Wind {int(wind)} mph")
            if outdoors and kind in WEATHER_HIT and pd.notna(ctx["temp"]) and ctx["temp"] <= 25:
                wx_f *= 0.97
                flags.append(f"Cold {int(ctx['temp'])}°F")

            proj = base * opp_f * env_f * script_f * wx_f

            # volatility across all recent games (this season + last)
            all_vals = pd.concat([g_prev[stat].tail(8), g_cur[stat]]).to_numpy(float)
            cv = np.std(all_vals) / max(np.mean(all_vals), 0.1) if len(all_vals) > 2 else np.nan

            st = status.get(pid)
            if st in ("Out", "Doubtful"):
                flags.append(f"{st}: {injury.get(pid, 'injury')}")
            elif st == "Questionable":
                flags.append(f"Questionable: {injury.get(pid, 'injury')}")
            if abs(ctx["margin"]) >= 9.5 if pd.notna(ctx["margin"]) else False:
                flags.append("Blowout risk")
            if p["team"] in vacated and kind != "volume_pass":
                others = [n for n in vacated[p["team"]] if n != p["player_display_name"]]
                if others:
                    flags.append(f"Vacated volume ({', '.join(others)} out)")
            if pd.notna(cv) and cv > 0.75 and kind not in ("volume_pass", "volume_rec"):
                flags.append("High volatility")

            conf = confidence(len(g_cur), len(g_prev), cv, flags)
            out.append(dict(
                player=p["player_display_name"], pos=p["pos_group"], team=p["team"], opp=ctx["opp"],
                matchup=ctx["matchup"], game_id=ctx["game_id"], market=market,
                projection=round(proj, 2), line=post_line(proj * LINE_SHADE[market], stat),
                confidence=conf, risk=", ".join(flags),
                status="Pull" if st in ("Out", "Doubtful") else ("Review" if flags else "Post"),
                base=round(base, 2), opp_adj=round(opp_f, 3), env_adj=round(env_f * script_f, 3),
                weather_adj=round(wx_f, 3), games_this_season=len(g_cur), player_id=pid,
            ))
    return pd.DataFrame(out)


def post_line(proj, stat):
    """PrizePicks-style lines end in .5 so there is no push. Round to the nearest x.5."""
    return max(round(proj - 0.5) + 0.5, 0.5)


def confidence(n_cur, n_prev, cv, flags):
    serious = [f for f in flags if not f.startswith(("Wind", "Cold"))]
    score = 0
    score += 2 if n_cur >= 3 else (1 if n_cur >= 2 else 0)
    score += 1 if n_prev >= 6 else 0
    score += 2 if (pd.notna(cv) and cv < 0.4) else (1 if (pd.notna(cv) and cv < 0.7) else 0)
    score -= len(serious)
    return "High" if score >= 4 else ("Medium" if score >= 2 else "Low")


# ---------------------------------------------------------------- backtest
def backtest(stats, games, season, weeks):
    """Project each past week using only data available before it, then grade against actuals."""
    empty_inj = pd.DataFrame(columns=["season", "week", "gsis_id", "report_status", "report_primary_injury"])
    rows = []
    for wk in weeks:
        proj = project_slate(stats, games, empty_inj, season, wk)
        if proj.empty:
            continue
        actual = stats[(stats["season"] == season) & (stats["week"] == wk)].set_index("player_id")
        hist = stats[(stats["season"] == season) & (stats["week"] < wk)]
        naive_by = hist.groupby("player_id")[[c["stat"] for c in MARKETS.values()]].mean()
        for _, r in proj.iterrows():
            if r["player_id"] not in actual.index:
                continue  # didn't play; a live book would void these
            stat = MARKETS[r["market"]]["stat"]
            a = actual.loc[r["player_id"], stat]
            a = float(a.iloc[0]) if isinstance(a, pd.Series) else float(a)
            naive = naive_by.loc[r["player_id"], stat] if r["player_id"] in naive_by.index else np.nan
            rows.append(dict(week=wk, market=r["market"], projection=r["projection"], line=r["line"],
                             actual=a, naive=naive, confidence=r["confidence"]))
    bt = pd.DataFrame(rows)
    if bt.empty:
        return bt, pd.DataFrame()
    bt["over"] = bt["actual"] > bt["line"]
    bt["err_model"] = (bt["projection"] - bt["actual"]).abs()
    bt["err_naive"] = (bt["naive"] - bt["actual"]).abs()
    summary = bt.dropna(subset=["naive"]).groupby("market").agg(
        props=("actual", "size"),
        model_mae=("err_model", "mean"),
        naive_mae=("err_naive", "mean"),
        over_rate=("over", "mean"),
    )
    summary["mae_improvement"] = 1 - summary["model_mae"] / summary["naive_mae"]
    return bt, summary
