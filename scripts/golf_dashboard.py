"""
Golf EV Dashboard — Streamlit app for visualising DataGolf analytics
and finding the best Kalshi golf bets.

Run:  streamlit run scripts/golf_dashboard.py
"""

import asyncio
import sys
import os

# Ensure project root is importable
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from src.clients.datagolf_client import DataGolfClient
from src.clients.kalshi_client import KalshiClient
from src.data.golf_data_fetcher import parse_golf_market, _to_first_last, _find_player_in_list
from src.utils.database import DatabaseManager

# ──────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────

def _run(coro):
    """Run an async coroutine from sync Streamlit context."""
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


@st.cache_data(ttl=300, show_spinner="Fetching Kalshi golf markets...")
def fetch_kalshi_markets():
    """Fetch all active golf markets from Kalshi."""

    async def _fetch():
        kalshi = KalshiClient()
        try:
            series_tickers = await kalshi.get_golf_series_tickers()
            all_markets = []
            for st_ticker in series_tickers:
                cursor = None
                while True:
                    resp = await kalshi.get_markets(
                        limit=100, cursor=cursor, series_ticker=st_ticker
                    )
                    page = resp.get("markets", [])
                    active = [m for m in page if m.get("status") == "active"]
                    all_markets.extend(active)
                    cursor = resp.get("cursor")
                    if not cursor:
                        break
            return all_markets
        finally:
            await kalshi.close()

    return _run(_fetch())


@st.cache_data(ttl=300, show_spinner="Fetching DataGolf data...")
def fetch_datagolf_data():
    """Fetch predictions, outright odds, decompositions, skill ratings, and field updates."""

    async def _fetch():
        dg = DataGolfClient()
        try:
            preds, odds, decomps, skills, field = await asyncio.gather(
                dg.get_pre_tournament_predictions(),
                dg.get_outright_odds(market="win"),
                dg.get_player_decompositions(),
                dg.get_skill_ratings(),
                dg.get_field_updates(),
            )
            return preds, odds, decomps, skills, field
        finally:
            await dg.close()

    return _run(_fetch())


@st.cache_data(ttl=300, show_spinner="Fetching live tournament data...")
def fetch_live_data():
    """Fetch live tournament stats (may be empty outside tournament hours)."""

    async def _fetch():
        dg = DataGolfClient()
        try:
            return await dg.get_live_tournament_stats()
        finally:
            await dg.close()

    return _run(_fetch())


@st.cache_data(ttl=3600, show_spinner="Fetching historical event list...")
def fetch_historical_event_list():
    """Fetch list of all historical tournaments from DataGolf."""

    async def _fetch():
        dg = DataGolfClient()
        try:
            return await dg.get_historical_event_list()
        finally:
            await dg.close()

    return _run(_fetch())


@st.cache_data(ttl=3600, show_spinner="Fetching course history rounds...")
def fetch_course_history(event_ids_tuple, years_tuple):
    """Fetch historical round data for matching events (hashable args for caching)."""

    async def _fetch():
        dg = DataGolfClient()
        try:
            tasks = [
                dg.get_historical_rounds(event_id=eid, year=yr)
                for eid, yr in zip(event_ids_tuple, years_tuple)
            ]
            results = await asyncio.gather(*tasks)
            return results
        finally:
            await dg.close()

    return _run(_fetch())


@st.cache_data(ttl=300, show_spinner="Fetching approach skill data...")
def fetch_approach_skill():
    """Fetch approach shot skill stats from DataGolf."""

    async def _fetch():
        dg = DataGolfClient()
        try:
            return await dg.get_approach_skill(period="l24")
        finally:
            await dg.close()

    return _run(_fetch())


def _normalize(name: str) -> str:
    """Lowercase and strip whitespace for fuzzy matching."""
    return name.strip().lower()


def find_matching_events(event_list, current_event_name, current_course_name, max_years=5):
    """Find past editions of the current tournament by event name or course name.

    Returns list of (event_id, calendar_year) tuples.
    """
    if not isinstance(event_list, list):
        return []

    norm_event = _normalize(current_event_name)
    norm_course = _normalize(current_course_name)

    # Collect matches: prefer event name match, fallback to course name match
    matches = []
    for evt in event_list:
        evt_name = _normalize(evt.get("event_name", ""))
        evt_course = _normalize(evt.get("course", evt.get("course_name", "")))
        if norm_event and norm_event in evt_name:
            matches.append((evt.get("event_id"), evt.get("calendar_year"), "event"))
        elif norm_course and norm_course in evt_course:
            matches.append((evt.get("event_id"), evt.get("calendar_year"), "course"))

    # Sort by year descending, take most recent N
    matches.sort(key=lambda x: x[1] or 0, reverse=True)
    matches = matches[:max_years]
    return [(eid, yr) for eid, yr, _ in matches]


@st.cache_resource
def get_db():
    """Initialise DatabaseManager once and cache the instance."""
    db = DatabaseManager()
    _run(db.initialize())
    return db


@st.cache_data(ttl=30, show_spinner="Fetching Kalshi positions...")
def fetch_kalshi_positions():
    """Fetch portfolio positions from Kalshi exchange."""

    async def _fetch():
        kalshi = KalshiClient()
        try:
            return await kalshi.get_positions()
        finally:
            await kalshi.close()

    return _run(_fetch())


@st.cache_data(ttl=30, show_spinner="Fetching market prices...")
def fetch_markets_by_tickers(tickers_tuple):
    """Batch-fetch market data for a tuple of tickers."""

    async def _fetch():
        kalshi = KalshiClient()
        try:
            tickers = list(tickers_tuple)
            if not tickers:
                return {}
            all_markets = {}
            for i in range(0, len(tickers), 50):
                chunk = tickers[i : i + 50]
                resp = await kalshi.get_markets(tickers=chunk, limit=200)
                for m in resp.get("markets", []):
                    all_markets[m["ticker"]] = m
            return all_markets
        finally:
            await kalshi.close()

    return _run(_fetch())


@st.cache_data(ttl=60, show_spinner="Loading open positions from database...")
def fetch_db_positions():
    """Fetch open positions from the local database."""
    db = get_db()
    positions = _run(db.get_open_positions())
    rows = []
    for p in positions:
        rows.append(
            {
                "market_id": p.market_id,
                "side": p.side,
                "entry_price": p.entry_price,
                "quantity": p.quantity,
                "live": p.live,
                "confidence": p.confidence,
                "timestamp": p.timestamp,
            }
        )
    return rows


@st.cache_data(ttl=60, show_spinner="Loading trade history...")
def fetch_trade_logs():
    """Fetch all completed trade logs from the database."""
    db = get_db()
    logs = _run(db.get_all_trade_logs())
    rows = []
    for t in logs:
        rows.append(
            {
                "market_id": t.market_id,
                "side": t.side,
                "entry_price": t.entry_price,
                "exit_price": t.exit_price,
                "quantity": t.quantity,
                "pnl": t.pnl,
                "entry_timestamp": t.entry_timestamp,
                "exit_timestamp": t.exit_timestamp,
                "strategy": t.strategy,
            }
        )
    return rows


@st.cache_data(ttl=60, show_spinner="Loading strategy performance...")
def fetch_strategy_performance():
    """Fetch per-strategy performance breakdown from the database."""
    db = get_db()
    return _run(db.get_performance_by_strategy())


def build_ev_dataframe(kalshi_markets, predictions, outright_odds):
    """Build a DataFrame matching Kalshi markets to DataGolf probabilities with EV."""
    pred_list = predictions.get(
        "baseline", predictions.get("data", predictions.get("players", []))
    )
    if not isinstance(pred_list, list):
        pred_list = []

    odds_list = outright_odds.get("data", outright_odds.get("odds", []))
    if not isinstance(odds_list, list):
        odds_list = []

    rows = []
    for mkt in kalshi_markets:
        title = mkt.get("title", "")
        parsed = parse_golf_market(title)
        if parsed["market_type"] == "unknown" or not parsed["players"]:
            continue

        player = parsed["players"][0]
        market_type = parsed["market_type"]

        # DG probability
        prob_keys = {
            "outright": ("win_prob", "win"),
            "top_n": (
                f"top_{parsed.get('top_n', 10)}_prob",
                f"top_{parsed.get('top_n', 10)}",
            ),
            "make_cut": ("make_cut_prob", "make_cut"),
        }
        keys = prob_keys.get(market_type, ("win_prob", "win"))

        dg_prob = None
        player_row = _find_player_in_list(player, pred_list)
        if player_row:
            for k in keys:
                v = player_row.get(k)
                if v is not None:
                    dg_prob = float(v)
                    break

        if dg_prob is None and market_type == "outright":
            odds_row = _find_player_in_list(player, odds_list)
            if odds_row:
                v = odds_row.get("datagolf_prob", odds_row.get("dg_prob"))
                if v is not None:
                    dg_prob = float(v)

        if dg_prob is None:
            continue

        yes_price = (mkt.get("yes_bid", 0) + mkt.get("yes_ask", 0)) / 2 / 100
        if yes_price <= 0:
            continue

        edge = dg_prob - yes_price
        ev_per_dollar = dg_prob - yes_price
        implied_odds = 1 / yes_price if yes_price > 0 else 0

        rows.append(
            {
                "Player": player,
                "Tournament": parsed.get("tournament", ""),
                "Type": market_type,
                "DG Win%": dg_prob,
                "Kalshi Price": yes_price,
                "Edge": edge,
                "EV/Dollar": ev_per_dollar,
                "Implied Odds": implied_odds,
                "Volume": int(mkt.get("volume", 0)),
                "Ticker": mkt.get("ticker", ""),
            }
        )

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("EV/Dollar", ascending=False).reset_index(drop=True)
        df.index += 1
        df.index.name = "Rank"
    return df


def build_win_dataframe(predictions, decompositions, skill_ratings):
    """Build a full-field DataFrame ranked by win probability, joining predictions,
    decompositions, and skill ratings on dg_id."""
    pred_list = predictions.get(
        "baseline", predictions.get("data", predictions.get("players", []))
    )
    if not isinstance(pred_list, list):
        return pd.DataFrame()

    decomp_list = decompositions.get("data", decompositions.get("players", []))
    if not isinstance(decomp_list, list):
        decomp_list = []
    decomp_by_id = {p.get("dg_id"): p for p in decomp_list if p.get("dg_id")}

    skill_list = skill_ratings.get("data", skill_ratings.get("players", []))
    if not isinstance(skill_list, list):
        skill_list = []
    skill_by_id = {p.get("dg_id"): p for p in skill_list if p.get("dg_id")}

    rows = []
    for p in pred_list:
        dg_id = p.get("dg_id")
        name = p.get("player_name", "Unknown")
        decomp = decomp_by_id.get(dg_id, {})
        skill = skill_by_id.get(dg_id, {})

        win = p.get("win_prob", p.get("win"))
        if win is None:
            continue

        rows.append({
            "Player": _to_first_last(name).title(),
            "Country": p.get("country", ""),
            "Win %": float(win),
            "Top 5": p.get("top_5_prob", p.get("top_5")),
            "Top 10": p.get("top_10_prob", p.get("top_10")),
            "Top 20": p.get("top_20_prob", p.get("top_20")),
            "Make Cut": p.get("make_cut_prob", p.get("make_cut")),
            "SG Total": skill.get("sg_total", decomp.get("sg_total")),
            "SG OTT": skill.get("sg_ott", decomp.get("sg_ott")),
            "SG APP": skill.get("sg_app", decomp.get("sg_app")),
            "SG ARG": skill.get("sg_arg", decomp.get("sg_arg")),
            "SG Putt": skill.get("sg_putt", decomp.get("sg_putt")),
            "Course Fit": decomp.get("total_fit_adjustment", decomp.get("total_course_history_adjustment")),
            "Fit: Accuracy": decomp.get("driving_accuracy_adjustment"),
            "Fit: Approach": decomp.get("cf_approach_comp"),
            "Fit: Short Game": decomp.get("cf_short_comp"),
            "Fit: Distance": decomp.get("driving_distance_adjustment"),
            "Std Dev": decomp.get("std_deviation"),
        })

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("Win %", ascending=False).reset_index(drop=True)
        df.index += 1
        df.index.name = "Rank"
    return df


def build_sg_dataframe(decompositions):
    """Build a DataFrame of SG components from DataGolf decompositions."""
    player_list = decompositions.get(
        "data", decompositions.get("players", [])
    )
    if not isinstance(player_list, list):
        return pd.DataFrame()

    rows = []
    for p in player_list:
        name = p.get("player_name", "Unknown")
        rows.append(
            {
                "Player": _to_first_last(name).title(),
                "SG OTT": p.get("sg_ott"),
                "SG APP": p.get("sg_app"),
                "SG ARG": p.get("sg_arg"),
                "SG Putt": p.get("sg_putt"),
                "SG T2G": p.get("sg_t2g"),
                "SG Total": p.get("sg_total"),
            }
        )
    return pd.DataFrame(rows).dropna(subset=["SG Total"])


def build_skill_dataframe(skill_ratings):
    """Build a DataFrame of skill ratings."""
    player_list = skill_ratings.get(
        "data", skill_ratings.get("players", [])
    )
    if not isinstance(player_list, list):
        return pd.DataFrame()

    rows = []
    for p in player_list:
        name = p.get("player_name", "Unknown")
        rows.append(
            {
                "Player": _to_first_last(name).title(),
                "Ranking": p.get("ranking"),
                "Skill Estimate": p.get("skill_estimate"),
                "Country": p.get("country", ""),
            }
        )
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("Ranking").reset_index(drop=True)
    return df


# ──────────────────────────────────────────────────────────────────────
# Streamlit App
# ──────────────────────────────────────────────────────────────────────

st.set_page_config(page_title="Golf EV Dashboard", layout="wide")
st.title("Golf EV Dashboard")

# Sidebar navigation
page = st.sidebar.radio(
    "Page",
    [
        "Win This Week",
        "EV Rankings",
        "SG Breakdown",
        "Odds Comparison",
        "Tournament Overview",
        "Course History",
        "Live Positions",
        "P&L History",
    ],
)

# Fetch golf data only for golf-specific pages (avoid slow DataGolf calls on other pages)
_golf_pages = {"Win This Week", "EV Rankings", "SG Breakdown", "Odds Comparison", "Tournament Overview", "Course History"}
if page in _golf_pages:
    kalshi_markets = fetch_kalshi_markets()
    predictions, outright_odds, decompositions, skill_ratings, field_updates = (
        fetch_datagolf_data()
    )

# ─── Page 0: Win This Week ──────────────────────────────────────────
if page == "Win This Week":
    st.header("Win This Week")

    event_name = field_updates.get("event_name", "Unknown Event")
    course_name = field_updates.get("course_name", field_updates.get("course", "N/A"))
    last_updated = field_updates.get("last_updated", "")

    st.caption(f"**{event_name}** — {course_name}")

    win_df = build_win_dataframe(predictions, decompositions, skill_ratings)
    if win_df.empty:
        st.warning("No prediction data available.")
    else:
        # Summary metrics
        favorite = win_df.iloc[0]
        field_size = len(win_df)
        top5_concentration = win_df.head(5)["Win %"].sum()

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Favorite", f"{favorite['Player']} ({favorite['Win %']:.1%})")
        c2.metric("Field Size", field_size)
        c3.metric("Top-5 Concentration", f"{top5_concentration:.1%}")
        c4.metric("Last Updated", last_updated or "N/A")

        # Course DNA — which attributes matter most this week
        st.subheader(f"Course DNA — {course_name}")
        fit_components = {
            "Driving Accuracy": "Fit: Accuracy",
            "Approach": "Fit: Approach",
            "Short Game": "Fit: Short Game",
            "Driving Distance": "Fit: Distance",
        }
        dna_rows = []
        for label, col in fit_components.items():
            if col in win_df.columns:
                vals = win_df[col].dropna()
                if not vals.empty:
                    dna_rows.append({
                        "Attribute": label,
                        "Spread (max-min)": vals.max() - vals.min(),
                        "Std Dev": vals.std(),
                        "Best Fit": f"{win_df.loc[vals.idxmax(), 'Player']} ({vals.max():+.3f})",
                        "Worst Fit": f"{win_df.loc[vals.idxmin(), 'Player']} ({vals.min():+.3f})",
                    })
        if dna_rows:
            dna_df = pd.DataFrame(dna_rows).sort_values("Spread (max-min)", ascending=False).reset_index(drop=True)
            dna_df.index += 1
            dna_df.index.name = "#"

            # Horizontal bar showing relative importance
            fig_dna = px.bar(
                dna_df,
                y="Attribute",
                x="Spread (max-min)",
                orientation="h",
                height=220,
                color="Spread (max-min)",
                color_continuous_scale="Oranges",
            )
            fig_dna.update_layout(
                xaxis_title="Field Spread (strokes)",
                yaxis_title="",
                coloraxis_showscale=False,
                yaxis=dict(categoryorder="total ascending"),
                margin=dict(t=10),
            )
            col_chart, col_table = st.columns([1, 1])
            with col_chart:
                st.plotly_chart(fig_dna, use_container_width=True)
            with col_table:
                st.dataframe(
                    dna_df.style.format({"Spread (max-min)": "{:.3f}", "Std Dev": "{:.3f}"}),
                    use_container_width=True,
                )
            st.caption(
                "Wider spread = this attribute creates more separation between players at this course. "
                "The top attribute is the biggest differentiator for the week."
            )

        # Main table — full field ranked by Win %
        st.subheader("Full Field Rankings")

        def _color_win(val):
            if isinstance(val, (int, float)):
                if val > 0:
                    return "background-color: rgba(0,200,0,0.15)"
                elif val < 0:
                    return "background-color: rgba(200,0,0,0.10)"
            return ""

        pct_cols = ["Win %", "Top 5", "Top 10", "Top 20", "Make Cut"]
        signed_cols = [
            "SG Total", "SG OTT", "SG APP", "SG ARG", "SG Putt",
            "Course Fit", "Fit: Accuracy", "Fit: Approach", "Fit: Short Game", "Fit: Distance",
        ]
        color_cols = [c for c in ["Win %", "SG Total", "Course Fit",
                                   "Fit: Accuracy", "Fit: Approach", "Fit: Short Game", "Fit: Distance"]
                      if c in win_df.columns]

        fmt = {}
        for c in pct_cols:
            if c in win_df.columns:
                fmt[c] = "{:.2%}"
        for c in signed_cols:
            if c in win_df.columns:
                fmt[c] = "{:+.3f}"
        if "Std Dev" in win_df.columns:
            fmt["Std Dev"] = "{:.2f}"

        styled_win = win_df.style.map(
            _color_win, subset=color_cols
        ).format(fmt)
        st.dataframe(styled_win, use_container_width=True, height=600)

        # Win probability bar chart — top 20
        st.subheader("Win Probability — Top 20")
        top20 = win_df.head(20).copy()
        top20_plot = top20[["Player", "Win %"]].iloc[::-1]  # reverse for horizontal bar
        fig_bar = px.bar(
            top20_plot,
            y="Player",
            x="Win %",
            orientation="h",
            height=max(450, len(top20_plot) * 28),
            color="Win %",
            color_continuous_scale="Greens",
        )
        fig_bar.update_layout(
            xaxis_title="Win Probability",
            xaxis_tickformat=".1%",
            yaxis_title="",
            coloraxis_showscale=False,
        )
        st.plotly_chart(fig_bar, use_container_width=True)

        # Course fit vs win % scatter
        scatter_df = win_df.dropna(subset=["Course Fit"]).copy()
        if not scatter_df.empty:
            st.subheader("Course Fit vs Win Probability")
            fig_scatter = px.scatter(
                scatter_df,
                x="Course Fit",
                y="Win %",
                hover_name="Player",
                color="Course Fit",
                color_continuous_scale=["red", "white", "green"],
                color_continuous_midpoint=0,
                size="Win %",
                size_max=18,
                height=500,
            )
            fig_scatter.update_layout(
                xaxis_title="Course Fit Adjustment (strokes)",
                yaxis_title="Win Probability",
                yaxis_tickformat=".1%",
            )
            st.plotly_chart(fig_scatter, use_container_width=True)
            st.caption(
                "Players in the upper-right have both high win probability and a positive course fit boost."
            )
        else:
            st.info("No course fit data available for scatter plot.")


# ─── Page 1: EV Rankings ────────────────────────────────────────────
elif page == "EV Rankings":
    st.header("EV Rankings")
    st.caption(
        "Kalshi golf markets ranked by expected value using DataGolf model probabilities."
    )

    df = build_ev_dataframe(kalshi_markets, predictions, outright_odds)
    if df.empty:
        st.warning("No markets matched to DataGolf data.")
    else:
        # Summary metrics
        pos_ev = (df["EV/Dollar"] > 0).sum()
        best_ev = df["EV/Dollar"].iloc[0] if len(df) > 0 else 0
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Markets", len(df))
        col2.metric("Positive EV", pos_ev)
        col3.metric("Best EV/Dollar", f"{best_ev:+.3f}")

        # Color-code by EV
        def _color_ev(val):
            if isinstance(val, (int, float)):
                if val > 0:
                    return "background-color: rgba(0,200,0,0.15)"
                elif val < 0:
                    return "background-color: rgba(200,0,0,0.10)"
            return ""

        styled = df.style.map(
            _color_ev, subset=["Edge", "EV/Dollar"]
        ).format(
            {
                "DG Win%": "{:.2%}",
                "Kalshi Price": "{:.2f}",
                "Edge": "{:+.2%}",
                "EV/Dollar": "{:+.3f}",
                "Implied Odds": "{:.1f}x",
                "Volume": "{:,}",
            }
        )
        st.dataframe(styled, use_container_width=True, height=600)


# ─── Page 2: SG Breakdown ───────────────────────────────────────────
elif page == "SG Breakdown":
    st.header("Strokes Gained Breakdown")

    sg_df = build_sg_dataframe(decompositions)
    if sg_df.empty:
        st.warning("No SG decomposition data available.")
    else:
        players = st.multiselect(
            "Select players to compare",
            options=sorted(sg_df["Player"].unique()),
            default=sorted(sg_df["Player"].unique())[:5],
        )

        if players:
            sel = sg_df[sg_df["Player"].isin(players)]

            # Radar chart
            st.subheader("Radar Chart")
            categories = ["SG OTT", "SG APP", "SG ARG", "SG Putt"]
            fig_radar = go.Figure()
            for _, row in sel.iterrows():
                vals = [row.get(c, 0) or 0 for c in categories]
                vals.append(vals[0])  # close the polygon
                fig_radar.add_trace(
                    go.Scatterpolar(
                        r=vals,
                        theta=categories + [categories[0]],
                        fill="toself",
                        name=row["Player"],
                    )
                )
            fig_radar.update_layout(
                polar=dict(radialaxis=dict(visible=True)),
                showlegend=True,
                height=500,
            )
            st.plotly_chart(fig_radar, use_container_width=True)

            # Grouped bar chart
            st.subheader("SG Components (Bar Chart)")
            bar_data = sel.melt(
                id_vars=["Player"],
                value_vars=categories,
                var_name="Component",
                value_name="SG",
            )
            fig_bar = px.bar(
                bar_data,
                x="Component",
                y="SG",
                color="Player",
                barmode="group",
                height=400,
            )
            st.plotly_chart(fig_bar, use_container_width=True)

        # Skill ratings table
        st.subheader("Skill Ratings (Full Field)")
        skill_df = build_skill_dataframe(skill_ratings)
        if not skill_df.empty:
            st.dataframe(skill_df, use_container_width=True, height=400)
        else:
            st.info("No skill ratings data available.")


# ─── Page 3: Odds Comparison ────────────────────────────────────────
elif page == "Odds Comparison":
    st.header("Odds Comparison — DataGolf vs Kalshi")

    df = build_ev_dataframe(kalshi_markets, predictions, outright_odds)
    df_outright = df[df["Type"] == "outright"].copy() if not df.empty else pd.DataFrame()

    if df_outright.empty:
        st.warning("No outright markets to compare.")
    else:
        # Top-20 grouped bar chart
        st.subheader("DG Model Prob vs Kalshi Implied Prob (Top 20)")
        top20 = df_outright.head(20)
        bar_data = pd.DataFrame(
            {
                "Player": top20["Player"].tolist() * 2,
                "Probability": top20["DG Win%"].tolist()
                + top20["Kalshi Price"].tolist(),
                "Source": ["DataGolf"] * len(top20) + ["Kalshi"] * len(top20),
            }
        )
        fig_bar = px.bar(
            bar_data,
            x="Player",
            y="Probability",
            color="Source",
            barmode="group",
            height=450,
        )
        fig_bar.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig_bar, use_container_width=True)

        # Scatter plot: DG prob vs Kalshi price
        st.subheader("Fair Value Scatter")
        fig_scatter = px.scatter(
            df_outright,
            x="DG Win%",
            y="Kalshi Price",
            hover_name="Player",
            color="Edge",
            color_continuous_scale=["red", "white", "green"],
            color_continuous_midpoint=0,
            size="Volume",
            size_max=20,
            height=500,
        )
        # Diagonal fair-value line
        max_val = max(
            df_outright["DG Win%"].max(), df_outright["Kalshi Price"].max(), 0.01
        )
        fig_scatter.add_trace(
            go.Scatter(
                x=[0, max_val],
                y=[0, max_val],
                mode="lines",
                line=dict(dash="dash", color="gray"),
                name="Fair Value",
                showlegend=True,
            )
        )
        fig_scatter.update_layout(
            xaxis_title="DataGolf Win Probability",
            yaxis_title="Kalshi YES Price",
        )
        st.plotly_chart(fig_scatter, use_container_width=True)

        st.caption(
            "Points below the line = underpriced on Kalshi (bet YES). "
            "Points above the line = overpriced on Kalshi (bet NO)."
        )

        # Biggest edge table
        st.subheader("Biggest Edges")
        edges = df_outright.head(10)[
            ["Player", "DG Win%", "Kalshi Price", "Edge", "EV/Dollar", "Volume"]
        ]
        st.dataframe(
            edges.style.format(
                {
                    "DG Win%": "{:.2%}",
                    "Kalshi Price": "{:.2f}",
                    "Edge": "{:+.2%}",
                    "EV/Dollar": "{:+.3f}",
                    "Volume": "{:,}",
                }
            ),
            use_container_width=True,
        )


# ─── Page 4: Tournament Overview ────────────────────────────────────
elif page == "Tournament Overview":
    st.header("Tournament Overview")

    # Tournament metadata
    event_name = field_updates.get("event_name", "Unknown Event")
    course = field_updates.get("course_name", field_updates.get("course", "N/A"))
    current_round = field_updates.get("current_round", "N/A")
    last_updated = field_updates.get("last_updated", "")

    col1, col2, col3 = st.columns(3)
    col1.metric("Event", event_name)
    col2.metric("Course", course)
    col3.metric("Round", current_round)

    if last_updated:
        st.caption(f"Last updated: {last_updated}")

    # Field list
    field_list = field_updates.get("data", field_updates.get("field", []))
    if isinstance(field_list, list) and field_list:
        st.subheader(f"Field ({len(field_list)} players)")

        wd_count = sum(
            1 for p in field_list if p.get("wd", False) or p.get("is_wd", False)
        )
        if wd_count:
            st.info(f"{wd_count} withdrawal(s)")

        field_rows = []
        for p in field_list:
            name = p.get("player_name", "Unknown")
            field_rows.append(
                {
                    "Player": _to_first_last(name).title(),
                    "DG Rank": p.get("dg_id", p.get("ranking", "")),
                    "Country": p.get("country", ""),
                    "WD": "Yes" if p.get("wd") or p.get("is_wd") else "",
                }
            )
        st.dataframe(pd.DataFrame(field_rows), use_container_width=True, height=500)
    else:
        st.info("No field data available yet.")

    # Live stats (if tournament is in progress)
    st.subheader("Live Stats")
    live_data = fetch_live_data()
    live_list = live_data.get("data", live_data.get("live_stats", []))
    if isinstance(live_list, list) and live_list:
        live_rows = []
        for p in live_list:
            name = p.get("player_name", "Unknown")
            live_rows.append(
                {
                    "Player": _to_first_last(name).title(),
                    "Position": p.get("position", ""),
                    "Total": p.get("total", ""),
                    "Thru": p.get("thru", ""),
                    "SG Total": p.get("sg_total", ""),
                    "SG OTT": p.get("sg_ott", ""),
                    "SG APP": p.get("sg_app", ""),
                    "SG ARG": p.get("sg_arg", ""),
                    "SG Putt": p.get("sg_putt", ""),
                }
            )
        st.dataframe(pd.DataFrame(live_rows), use_container_width=True, height=500)
    else:
        st.info("No live tournament data available (tournament may not be in progress).")


# ─── Page 5: Course History ────────────────────────────────────────
elif page == "Course History":
    st.header("Course History")

    event_name = field_updates.get("event_name", "Unknown Event")
    course_name = field_updates.get("course_name", field_updates.get("course", "N/A"))

    # Fetch historical event list and find matching past editions
    hist_events_raw = fetch_historical_event_list()
    hist_event_list = hist_events_raw.get("data", hist_events_raw.get("events", []))
    if isinstance(hist_event_list, list) and isinstance(hist_event_list[0:1], list):
        # Handle case where API returns nested list
        pass

    matched = find_matching_events(hist_event_list, event_name, course_name)

    st.caption(f"**{event_name}** — {course_name}")

    if not matched:
        st.warning("No historical data found for this course/event.")
    else:
        years_found = [yr for _, yr in matched]
        st.info(f"Found {len(matched)} past edition(s): {', '.join(str(y) for y in years_found)}")

        # Fetch round-level data for all matching events
        event_ids_tuple = tuple(eid for eid, _ in matched)
        years_tuple = tuple(yr for _, yr in matched)
        round_results = fetch_course_history(event_ids_tuple, years_tuple)

        # Build current field set for filtering
        field_list = field_updates.get("data", field_updates.get("field", []))
        field_names = set()
        if isinstance(field_list, list):
            for p in field_list:
                name = p.get("player_name", "")
                field_names.add(_normalize(name))
                field_names.add(_normalize(_to_first_last(name)))

        # Aggregate round data per player
        player_rounds = {}  # player_name -> list of round dicts
        for result in round_results:
            rounds_list = result.get("data", result.get("rounds", []))
            if not isinstance(rounds_list, list):
                continue
            for r in rounds_list:
                pname = r.get("player_name", "")
                norm = _normalize(pname)
                norm_fl = _normalize(_to_first_last(pname))
                if norm not in field_names and norm_fl not in field_names:
                    continue
                display_name = _to_first_last(pname).title()
                player_rounds.setdefault(display_name, []).append(r)

        if not player_rounds:
            st.warning("No current field players found in historical data for this course.")
        else:
            # Build aggregated stats
            agg_rows = []
            for player, rounds in player_rounds.items():
                sg_totals = [r.get("sg_total") for r in rounds if r.get("sg_total") is not None]
                sg_otts = [r.get("sg_ott") for r in rounds if r.get("sg_ott") is not None]
                sg_apps = [r.get("sg_app") for r in rounds if r.get("sg_app") is not None]
                sg_args = [r.get("sg_arg") for r in rounds if r.get("sg_arg") is not None]
                sg_putts = [r.get("sg_putt") for r in rounds if r.get("sg_putt") is not None]
                scores = [r.get("score") for r in rounds if r.get("score") is not None]

                if not sg_totals:
                    continue

                avg_total = sum(sg_totals) / len(sg_totals)
                agg_rows.append({
                    "Player": player,
                    "Rounds": len(sg_totals),
                    "Avg SG Total": round(avg_total, 2),
                    "SG OTT": round(sum(sg_otts) / len(sg_otts), 2) if sg_otts else None,
                    "SG APP": round(sum(sg_apps) / len(sg_apps), 2) if sg_apps else None,
                    "SG ARG": round(sum(sg_args) / len(sg_args), 2) if sg_args else None,
                    "SG Putt": round(sum(sg_putts) / len(sg_putts), 2) if sg_putts else None,
                    "Best Round": round(max(sg_totals), 2),
                    "Worst Round": round(min(sg_totals), 2),
                })

            hist_df = pd.DataFrame(agg_rows)
            if hist_df.empty:
                st.warning("No SG data available for current field players at this course.")
            else:
                hist_df = hist_df.sort_values("Avg SG Total", ascending=False).reset_index(drop=True)

                # Summary metrics
                col1, col2, col3 = st.columns(3)
                col1.metric("Field Avg Course SG", f"{hist_df['Avg SG Total'].mean():+.2f}")
                col2.metric("Players w/ Course History", len(hist_df))
                col3.metric("Total Rounds in Dataset", int(hist_df["Rounds"].sum()))

                # Course History table
                st.subheader("Course History — Current Field")

                def _color_sg(val):
                    if isinstance(val, (int, float)):
                        if val > 0:
                            return "background-color: rgba(0,200,0,0.15)"
                        elif val < 0:
                            return "background-color: rgba(200,0,0,0.10)"
                    return ""

                sg_cols = ["Avg SG Total", "SG OTT", "SG APP", "SG ARG", "SG Putt", "Best Round", "Worst Round"]
                styled_hist = hist_df.style.map(
                    _color_sg, subset=[c for c in sg_cols if c in hist_df.columns]
                ).format(
                    {c: "{:+.2f}" for c in sg_cols if c in hist_df.columns}
                )
                st.dataframe(styled_hist, use_container_width=True, height=500)

                # Horizontal stacked bar chart — top 15 by avg SG Total
                st.subheader("Top 15 — Course SG Breakdown")
                top15 = hist_df.head(15).copy()
                sg_components = ["SG OTT", "SG APP", "SG ARG", "SG Putt"]
                chart_data = top15[["Player"] + [c for c in sg_components if c in top15.columns]].copy()
                chart_melted = chart_data.melt(
                    id_vars=["Player"], var_name="Component", value_name="SG"
                )
                fig_bar = px.bar(
                    chart_melted,
                    y="Player",
                    x="SG",
                    color="Component",
                    orientation="h",
                    height=max(400, len(top15) * 35),
                    color_discrete_map={
                        "SG OTT": "#636EFA",
                        "SG APP": "#EF553B",
                        "SG ARG": "#00CC96",
                        "SG Putt": "#AB63FA",
                    },
                )
                fig_bar.update_layout(
                    yaxis=dict(categoryorder="total ascending"),
                    xaxis_title="Strokes Gained",
                    yaxis_title="",
                    barmode="relative",
                )
                st.plotly_chart(fig_bar, use_container_width=True)

    # Build field name set for approach skill filtering
    if "field_names" not in dir() or not field_names:
        field_list = field_updates.get("data", field_updates.get("field", []))
        field_names = set()
        if isinstance(field_list, list):
            for p in field_list:
                name = p.get("player_name", "")
                field_names.add(_normalize(name))
                field_names.add(_normalize(_to_first_last(name)))

    # Approach Skill section (always shown)
    st.subheader("Approach Skill (Last 24 Months)")
    approach_data = fetch_approach_skill()
    approach_list = approach_data.get("data", approach_data.get("players", []))
    if isinstance(approach_list, list) and approach_list:
        # Distance buckets from the API (fairway and rough)
        _fw_buckets = ["50_100_fw", "100_150_fw", "150_200_fw", "over_200_fw"]
        _rgh_buckets = ["under_150_rgh", "over_150_rgh"]

        app_rows = []
        for p in approach_list:
            pname = p.get("player_name", "")
            norm = _normalize(pname)
            norm_fl = _normalize(_to_first_last(pname))
            if field_names and norm not in field_names and norm_fl not in field_names:
                continue

            # Weighted-average SG/shot across all buckets
            total_sg, total_shots = 0.0, 0
            for bucket in _fw_buckets + _rgh_buckets:
                sg = p.get(f"{bucket}_sg_per_shot")
                shots = p.get(f"{bucket}_shot_count")
                if sg is not None and shots is not None and shots > 0:
                    total_sg += sg * shots
                    total_shots += shots
            avg_sg = round(total_sg / total_shots, 4) if total_shots > 0 else None

            # Key per-bucket stats
            prox_150_200 = p.get("150_200_fw_proximity_per_shot")
            gir_150_200 = p.get("150_200_fw_gir_rate")
            sg_150_200 = p.get("150_200_fw_sg_per_shot")
            prox_200 = p.get("over_200_fw_proximity_per_shot")
            gir_100_150 = p.get("100_150_fw_gir_rate")
            sg_100_150 = p.get("100_150_fw_sg_per_shot")

            app_rows.append({
                "Player": _to_first_last(pname).title(),
                "Wtd SG/Shot": avg_sg,
                "Shots": total_shots,
                "SG 100-150": sg_100_150,
                "GIR 100-150": gir_100_150,
                "SG 150-200": sg_150_200,
                "GIR 150-200": gir_150_200,
                "Prox 150-200": prox_150_200,
                "Prox 200+": prox_200,
            })

        app_df = pd.DataFrame(app_rows)
        app_df = app_df.dropna(subset=["Wtd SG/Shot"])
        if not app_df.empty:
            app_df = app_df.sort_values("Wtd SG/Shot", ascending=False).reset_index(drop=True)

            def _color_app_sg(val):
                if isinstance(val, (int, float)):
                    if val > 0:
                        return "background-color: rgba(0,200,0,0.15)"
                    elif val < 0:
                        return "background-color: rgba(200,0,0,0.10)"
                return ""

            sg_app_cols = [c for c in ["Wtd SG/Shot", "SG 100-150", "SG 150-200"] if c in app_df.columns]
            fmt = {
                "Wtd SG/Shot": "{:+.4f}",
                "SG 100-150": "{:+.3f}",
                "SG 150-200": "{:+.3f}",
                "GIR 100-150": "{:.1%}",
                "GIR 150-200": "{:.1%}",
                "Prox 150-200": "{:.1f} ft",
                "Prox 200+": "{:.1f} ft",
            }
            styled_app = app_df.style.map(
                _color_app_sg, subset=sg_app_cols
            ).format({k: v for k, v in fmt.items() if k in app_df.columns})
            st.dataframe(styled_app, use_container_width=True, height=400)
        else:
            st.info("No approach skill data available for current field.")
    else:
        st.info("No approach skill data available.")


# ─── Page 6: Live Positions ────────────────────────────────────────
elif page == "Live Positions":
    st.header("Live Positions")
    st.caption("Open positions from Kalshi exchange and paper trading.")

    kalshi_data = fetch_kalshi_positions()
    db_pos_rows = fetch_db_positions()

    kalshi_pos_list = kalshi_data.get("market_positions", [])
    active_kalshi = [p for p in kalshi_pos_list if p.get("position", 0) != 0]

    # Collect all tickers that need market data
    all_tickers = set()
    for p in active_kalshi:
        all_tickers.add(p["ticker"])
    for p in db_pos_rows:
        if not p["live"]:
            all_tickers.add(p["market_id"])

    # Batch-fetch current market data
    market_details = (
        fetch_markets_by_tickers(tuple(sorted(all_tickers))) if all_tickers else {}
    )

    rows = []

    # --- Kalshi exchange positions ---
    for p in active_kalshi:
        ticker = p["ticker"]
        position = p.get("position", 0)
        side = "YES" if position > 0 else "NO"
        qty = abs(position)

        # Estimate entry price from market_exposure (cents)
        exposure = abs(p.get("market_exposure", 0))
        entry_price = (exposure / qty / 100) if qty > 0 and exposure > 0 else 0

        mkt = market_details.get(ticker, {})
        title = mkt.get("title", ticker)

        if side == "YES":
            bid, ask = mkt.get("yes_bid", 0), mkt.get("yes_ask", 0)
        else:
            bid, ask = mkt.get("no_bid", 0), mkt.get("no_ask", 0)
        current_price = (bid + ask) / 2 / 100 if (bid + ask) > 0 else 0

        unrealized = (current_price - entry_price) * qty
        pnl_pct = (
            (unrealized / (entry_price * qty) * 100)
            if entry_price * qty > 0
            else 0
        )

        rows.append(
            {
                "Ticker": ticker,
                "Title": title,
                "Side": side,
                "Qty": qty,
                "Entry Price": entry_price,
                "Current Price": current_price,
                "Unrealized P&L": unrealized,
                "P&L %": pnl_pct,
                "Source": "Live",
            }
        )

    # --- Paper positions (DB, not live) ---
    for p in db_pos_rows:
        if p["live"]:
            continue  # Already covered by Kalshi API
        ticker = p["market_id"]
        side = p["side"]
        qty = p["quantity"]
        entry_price = p["entry_price"]

        mkt = market_details.get(ticker, {})
        title = mkt.get("title", ticker)

        if side == "YES":
            bid, ask = mkt.get("yes_bid", 0), mkt.get("yes_ask", 0)
        else:
            bid, ask = mkt.get("no_bid", 0), mkt.get("no_ask", 0)
        current_price = (bid + ask) / 2 / 100 if (bid + ask) > 0 else 0

        unrealized = (current_price - entry_price) * qty
        pnl_pct = (
            (unrealized / (entry_price * qty) * 100)
            if entry_price * qty > 0
            else 0
        )

        rows.append(
            {
                "Ticker": ticker,
                "Title": title,
                "Side": side,
                "Qty": qty,
                "Entry Price": entry_price,
                "Current Price": current_price,
                "Unrealized P&L": unrealized,
                "P&L %": pnl_pct,
                "Source": "Paper",
            }
        )

    if not rows:
        st.info("No open positions.")
    else:
        df = pd.DataFrame(rows)

        # Summary metrics
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Positions", len(df))
        total_exposure = (df["Qty"] * df["Entry Price"]).sum()
        col2.metric("Total Exposure", f"${total_exposure:,.2f}")
        total_upnl = df["Unrealized P&L"].sum()
        col3.metric("Unrealized P&L", f"${total_upnl:+,.2f}")

        # Color-code P&L columns
        def _color_pnl(val):
            if isinstance(val, (int, float)):
                if val > 0:
                    return "background-color: rgba(0,200,0,0.15)"
                elif val < 0:
                    return "background-color: rgba(200,0,0,0.10)"
            return ""

        styled = df.style.map(
            _color_pnl, subset=["Unrealized P&L", "P&L %"]
        ).format(
            {
                "Entry Price": "${:.2f}",
                "Current Price": "${:.2f}",
                "Unrealized P&L": "${:+,.2f}",
                "P&L %": "{:+.1f}%",
            }
        )
        st.dataframe(styled, use_container_width=True, height=500)


# ─── Page 7: P&L History ───────────────────────────────────────────
elif page == "P&L History":
    st.header("P&L History")
    st.caption("Completed trades, cumulative P&L, and strategy breakdown.")

    trade_rows = fetch_trade_logs()
    strategy_perf = fetch_strategy_performance()

    if not trade_rows:
        st.info("No completed trades yet.")
    else:
        df = pd.DataFrame(trade_rows)
        df = df.sort_values("exit_timestamp").reset_index(drop=True)

        # ── Summary metrics ──
        total_pnl = df["pnl"].sum()
        total_trades = len(df)
        wins = (df["pnl"] > 0).sum()
        win_rate = wins / total_trades * 100 if total_trades > 0 else 0
        avg_pnl = df["pnl"].mean()

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Realized P&L", f"${total_pnl:+,.2f}")
        c2.metric("Win Rate", f"{win_rate:.1f}%")
        c3.metric("Total Trades", total_trades)
        c4.metric("Avg P&L / Trade", f"${avg_pnl:+,.2f}")

        # ── Cumulative P&L line chart ──
        st.subheader("Cumulative P&L")
        df["cumulative_pnl"] = df["pnl"].cumsum()
        fig_cum = px.line(
            df,
            x="exit_timestamp",
            y="cumulative_pnl",
            labels={
                "exit_timestamp": "Date",
                "cumulative_pnl": "Cumulative P&L ($)",
            },
            height=400,
        )
        fig_cum.add_hline(y=0, line_dash="dash", line_color="gray")
        st.plotly_chart(fig_cum, use_container_width=True)

        # ── Per-trade table ──
        st.subheader("Trade Log")
        display_df = df[
            [
                "market_id",
                "side",
                "entry_price",
                "exit_price",
                "quantity",
                "pnl",
                "strategy",
                "exit_timestamp",
            ]
        ].copy()
        display_df.rename(
            columns={
                "market_id": "Ticker",
                "side": "Side",
                "entry_price": "Entry",
                "exit_price": "Exit",
                "quantity": "Qty",
                "pnl": "P&L ($)",
                "strategy": "Strategy",
                "exit_timestamp": "Date",
            },
            inplace=True,
        )
        display_df["P&L (%)"] = display_df.apply(
            lambda r: (r["P&L ($)"] / (r["Entry"] * r["Qty"]) * 100)
            if r["Entry"] * r["Qty"] > 0
            else 0,
            axis=1,
        )

        def _color_trade_pnl(val):
            if isinstance(val, (int, float)):
                if val > 0:
                    return "background-color: rgba(0,200,0,0.15)"
                elif val < 0:
                    return "background-color: rgba(200,0,0,0.10)"
            return ""

        styled_trades = display_df.style.map(
            _color_trade_pnl, subset=["P&L ($)", "P&L (%)"]
        ).format(
            {
                "Entry": "${:.2f}",
                "Exit": "${:.2f}",
                "P&L ($)": "${:+,.2f}",
                "P&L (%)": "{:+.1f}%",
            }
        )
        st.dataframe(styled_trades, use_container_width=True, height=400)

        # ── Strategy breakdown ──
        if strategy_perf:
            st.subheader("Strategy Breakdown")
            strat_rows = []
            for name, stats in strategy_perf.items():
                strat_rows.append(
                    {
                        "Strategy": name,
                        "Trades": stats.get("completed_trades", 0),
                        "Total P&L": stats.get("total_pnl", 0),
                        "Win Rate": stats.get("win_rate_pct", 0),
                        "Avg P&L": stats.get("avg_pnl_per_trade", 0),
                        "Best Trade": stats.get("best_trade", 0),
                        "Worst Trade": stats.get("worst_trade", 0),
                    }
                )
            strat_df = pd.DataFrame(strat_rows)
            if not strat_df.empty:
                styled_strat = strat_df.style.format(
                    {
                        "Total P&L": "${:+,.2f}",
                        "Win Rate": "{:.1f}%",
                        "Avg P&L": "${:+,.2f}",
                        "Best Trade": "${:+,.2f}",
                        "Worst Trade": "${:+,.2f}",
                    }
                )
                st.dataframe(styled_strat, use_container_width=True)

        # ── P&L distribution histogram ──
        st.subheader("P&L Distribution")
        fig_hist = px.histogram(
            df,
            x="pnl",
            nbins=30,
            labels={"pnl": "P&L ($)"},
            height=350,
        )
        fig_hist.add_vline(x=0, line_dash="dash", line_color="gray")
        st.plotly_chart(fig_hist, use_container_width=True)
