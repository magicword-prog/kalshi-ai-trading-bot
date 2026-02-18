"""
Golf Data Fetcher — orchestrates DataGolf API calls and formats data for AI consumption.

Parses Kalshi golf market titles to identify the market type (outright, top-N,
matchup, make-cut) and the player(s) involved, then fetches relevant DataGolf
analytics and returns a structured context dict for the AI agents.
"""

import asyncio
import re
from typing import Any, Dict, List, Optional, Tuple

from src.clients.datagolf_client import DataGolfClient
from src.utils.logging_setup import get_trading_logger

logger = get_trading_logger("golf_data_fetcher")

# ---------------------------------------------------------------------------
# Market title parsing patterns (Kalshi PGA markets)
# ---------------------------------------------------------------------------
# "Will Scottie Scheffler win the Masters?"
_OUTRIGHT_RE = re.compile(
    r"Will\s+(.+?)\s+win\s+(?:the\s+)?(.+?)\??$", re.IGNORECASE
)
# "Will Scottie Scheffler finish top 10 at the Masters?"
_TOP_N_RE = re.compile(
    r"Will\s+(.+?)\s+finish\s+top\s+(\d+)\s+(?:at\s+)?(?:the\s+)?(.+?)\??$",
    re.IGNORECASE,
)
# "Scottie Scheffler vs Rory McIlroy at the Masters"
_MATCHUP_RE = re.compile(
    r"(.+?)\s+vs\.?\s+(.+?)\s+at\s+(?:the\s+)?(.+?)$", re.IGNORECASE
)
# "Will Scottie Scheffler make the cut at the Masters?"
_MAKE_CUT_RE = re.compile(
    r"Will\s+(.+?)\s+make\s+the\s+cut\s+(?:at\s+)?(?:the\s+)?(.+?)\??$",
    re.IGNORECASE,
)


def parse_golf_market(title: str) -> Dict[str, Any]:
    """
    Parse a Kalshi golf market title into structured components.

    Returns dict with keys:
        market_type: "outright" | "top_n" | "matchup" | "make_cut" | "unknown"
        players:     list of player name strings
        tournament:  tournament name (if parsed)
        top_n:       int (only for top_n markets)
    """
    # Try each pattern in order of specificity
    m = _TOP_N_RE.match(title)
    if m:
        return {
            "market_type": "top_n",
            "players": [m.group(1).strip()],
            "tournament": m.group(3).strip(),
            "top_n": int(m.group(2)),
        }

    m = _MAKE_CUT_RE.match(title)
    if m:
        return {
            "market_type": "make_cut",
            "players": [m.group(1).strip()],
            "tournament": m.group(2).strip(),
        }

    m = _OUTRIGHT_RE.match(title)
    if m:
        return {
            "market_type": "outright",
            "players": [m.group(1).strip()],
            "tournament": m.group(2).strip(),
        }

    m = _MATCHUP_RE.match(title)
    if m:
        return {
            "market_type": "matchup",
            "players": [m.group(1).strip(), m.group(2).strip()],
            "tournament": m.group(3).strip(),
        }

    return {"market_type": "unknown", "players": [], "tournament": ""}


def _normalize_name(name: str) -> str:
    """Lowercase, strip whitespace/punctuation for fuzzy matching."""
    return re.sub(r"[^a-z ]", "", name.lower()).strip()


def _to_first_last(name: str) -> str:
    """Convert 'Last, First' to 'first last' for comparison."""
    norm = _normalize_name(name)
    if "," in name:
        parts = name.split(",", 1)
        norm = _normalize_name(f"{parts[1].strip()} {parts[0].strip()}")
    return norm


def _find_player_in_list(
    target: str, player_list: List[Dict], name_key: str = "player_name"
) -> Optional[Dict]:
    """Find a player dict by fuzzy name match (handles 'Last, First' format)."""
    norm_target = _to_first_last(target)
    target_parts = norm_target.split()

    for p in player_list:
        raw = p.get(name_key, "")
        norm_raw = _to_first_last(raw)

        # Exact match
        if norm_raw == norm_target:
            return p

        # Last-name + first-initial match
        raw_parts = norm_raw.split()
        if target_parts and raw_parts:
            if (
                target_parts[-1] == raw_parts[-1]
                and target_parts[0][0] == raw_parts[0][0]
            ):
                return p
    return None


class GolfDataFetcher:
    """Orchestrates DataGolf API calls and formats data for AI agents."""

    def __init__(self, client: Optional[DataGolfClient] = None):
        self._client = client or DataGolfClient()

    async def get_golf_context(
        self, market_title: str, market_ticker: str
    ) -> Dict[str, Any]:
        """
        Parse the market title to identify player(s)/matchup/prop,
        then fetch relevant DataGolf data.

        Returns dict with:
            - market_parsed: parsed market info (type, players, tournament)
            - player_stats: SG breakdown, skill rating, recent form
            - tournament_info: field, course, conditions
            - model_predictions: DG win/top-5/top-10 probabilities
            - dg_odds: DataGolf model odds vs market price (edge signal)
            - matchup_data: head-to-head stats if matchup market
        """
        parsed = parse_golf_market(market_title)
        logger.info(
            "Parsed golf market",
            title=market_title,
            type=parsed["market_type"],
            players=parsed["players"],
        )

        context: Dict[str, Any] = {"market_parsed": parsed}

        if parsed["market_type"] == "unknown":
            logger.warning("Could not parse golf market title", title=market_title)
            return context

        # Fetch data in parallel
        tasks = {
            "predictions": self._client.get_pre_tournament_predictions(),
            "decompositions": self._client.get_player_decompositions(),
            "skill_ratings": self._client.get_skill_ratings(),
            "field": self._client.get_field_updates(),
        }

        # Add market-type specific fetches
        if parsed["market_type"] == "matchup":
            tasks["matchups"] = self._client.get_matchup_odds()
        if parsed["market_type"] == "outright":
            tasks["outrights"] = self._client.get_outright_odds(market="win")
        elif parsed["market_type"] == "top_n":
            top_n = parsed.get("top_n", 10)
            market_map = {5: "top_5", 10: "top_10", 20: "top_20"}
            dg_market = market_map.get(top_n, "top_10")
            tasks["outrights"] = self._client.get_outright_odds(market=dg_market)
        elif parsed["market_type"] == "make_cut":
            tasks["outrights"] = self._client.get_outright_odds(market="make_cut")

        # Also try live data (may not be available outside tournament hours)
        tasks["live_stats"] = self._client.get_live_tournament_stats()
        tasks["in_play"] = self._client.get_in_play_predictions()

        results = {}
        gathered = await asyncio.gather(
            *[tasks[k] for k in tasks], return_exceptions=True
        )
        for key, result in zip(tasks.keys(), gathered):
            if isinstance(result, Exception):
                logger.warning(f"DataGolf fetch failed for {key}", error=str(result))
                results[key] = {}
            else:
                results[key] = result

        # Extract per-player data
        players_data = {}
        for player_name in parsed["players"]:
            pdata = self._extract_player_data(player_name, results)
            players_data[player_name] = pdata

        context["player_stats"] = players_data
        context["model_predictions"] = self._extract_predictions(
            parsed["players"], results.get("predictions", {})
        )
        context["dg_odds"] = self._extract_odds(
            parsed["players"], results.get("outrights", {})
        )
        context["tournament_info"] = self._extract_tournament_info(results)

        if parsed["market_type"] == "matchup" and len(parsed["players"]) == 2:
            context["matchup_data"] = self._extract_matchup(
                parsed["players"], results.get("matchups", {})
            )

        # Live data (may be empty outside tournament play)
        live = self._extract_live_data(parsed["players"], results)
        if live:
            context["live_data"] = live

        return context

    # ------------------------------------------------------------------
    # Data extraction helpers
    # ------------------------------------------------------------------
    def _extract_player_data(
        self, player_name: str, results: Dict
    ) -> Dict[str, Any]:
        """Extract SG decomposition and skill rating for a player."""
        data: Dict[str, Any] = {}

        # Skill decompositions (SG breakdown)
        decomps = results.get("decompositions", {})
        player_list = decomps.get("data", decomps.get("players", []))
        if isinstance(player_list, list):
            player = _find_player_in_list(player_name, player_list)
            if player:
                data["sg_breakdown"] = {
                    "sg_ott": player.get("sg_ott"),
                    "sg_app": player.get("sg_app"),
                    "sg_arg": player.get("sg_arg"),
                    "sg_putt": player.get("sg_putt"),
                    "sg_t2g": player.get("sg_t2g"),
                    "sg_total": player.get("sg_total"),
                    "driving_distance": player.get("driving_dist"),
                    "driving_accuracy": player.get("driving_acc"),
                }

        # Skill ratings (also has SG data)
        ratings = results.get("skill_ratings", {})
        ratings_list = ratings.get("data", ratings.get("players", []))
        if isinstance(ratings_list, list):
            player = _find_player_in_list(player_name, ratings_list)
            if player:
                data["skill_rating"] = {
                    "skill_estimate": player.get("skill_estimate"),
                    "ranking": player.get("ranking"),
                    "country": player.get("country"),
                }
                # Fill SG breakdown from skill ratings if decompositions were empty
                sg = data.get("sg_breakdown", {})
                if not any(v is not None for v in sg.values()):
                    data["sg_breakdown"] = {
                        "sg_ott": player.get("sg_ott"),
                        "sg_app": player.get("sg_app"),
                        "sg_arg": player.get("sg_arg"),
                        "sg_putt": player.get("sg_putt"),
                        "sg_t2g": player.get("sg_t2g"),
                        "sg_total": player.get("sg_total"),
                        "driving_distance": player.get("driving_dist"),
                        "driving_accuracy": player.get("driving_acc"),
                    }

        return data

    def _extract_predictions(
        self, players: List[str], predictions: Dict
    ) -> Dict[str, Any]:
        """Extract DG model predictions for target players."""
        result = {}
        pred_list = predictions.get("baseline", predictions.get("data", predictions.get("players", [])))
        if not isinstance(pred_list, list):
            return result

        for player_name in players:
            player = _find_player_in_list(player_name, pred_list)
            if player:
                result[player_name] = {
                    "win": player.get("win_prob", player.get("win")),
                    "top_5": player.get("top_5_prob", player.get("top_5")),
                    "top_10": player.get("top_10_prob", player.get("top_10")),
                    "top_20": player.get("top_20_prob", player.get("top_20")),
                    "make_cut": player.get("make_cut_prob", player.get("make_cut")),
                }
        return result

    def _extract_odds(
        self, players: List[str], outrights: Dict
    ) -> Dict[str, Any]:
        """Extract DG model odds for target players."""
        result = {}
        odds_list = outrights.get("data", outrights.get("odds", []))
        if not isinstance(odds_list, list):
            return result

        for player_name in players:
            player = _find_player_in_list(player_name, odds_list)
            if player:
                result[player_name] = {
                    "dg_prob": player.get("datagolf_prob", player.get("dg_prob")),
                    "baseline_history_fit": player.get("baseline_history_fit"),
                    "dg_odds": player.get("datagolf"),
                }
        return result

    def _extract_tournament_info(self, results: Dict) -> Dict[str, Any]:
        """Extract tournament-level information from field updates."""
        field = results.get("field", {})
        info: Dict[str, Any] = {}

        # Tournament metadata (varies by DG response format)
        for key in ("event_name", "course", "current_round", "last_updated"):
            if key in field:
                info[key] = field[key]

        # Field size
        field_list = field.get("data", field.get("field", []))
        if isinstance(field_list, list):
            info["field_size"] = len(field_list)
            # Count WDs
            wds = [p for p in field_list if p.get("wd", False) or p.get("is_wd", False)]
            if wds:
                info["withdrawals"] = len(wds)

        return info

    def _extract_matchup(
        self, players: List[str], matchups: Dict
    ) -> Dict[str, Any]:
        """Extract head-to-head matchup data for two players."""
        matchup_list = matchups.get("data", matchups.get("matchups", []))
        if not isinstance(matchup_list, list) or len(players) < 2:
            return {}

        p1_norm = _normalize_name(players[0])
        p2_norm = _normalize_name(players[1])

        for m in matchup_list:
            m_p1 = _normalize_name(m.get("player_1", m.get("p1_name", "")))
            m_p2 = _normalize_name(m.get("player_2", m.get("p2_name", "")))
            if (p1_norm in m_p1 and p2_norm in m_p2) or (
                p2_norm in m_p1 and p1_norm in m_p2
            ):
                return {
                    "player_1": m.get("player_1", m.get("p1_name")),
                    "player_2": m.get("player_2", m.get("p2_name")),
                    "p1_win_prob": m.get("p1_prob", m.get("player_1_prob")),
                    "p2_win_prob": m.get("p2_prob", m.get("player_2_prob")),
                    "tie_prob": m.get("tie_prob"),
                }
        return {}

    def _extract_live_data(
        self, players: List[str], results: Dict
    ) -> Dict[str, Any]:
        """Extract live in-play data if available."""
        live: Dict[str, Any] = {}

        # Live stats
        live_stats = results.get("live_stats", {})
        stats_list = live_stats.get("data", live_stats.get("live_stats", []))
        if isinstance(stats_list, list):
            for player_name in players:
                player = _find_player_in_list(player_name, stats_list)
                if player:
                    live.setdefault(player_name, {})["live_sg"] = {
                        k: v
                        for k, v in player.items()
                        if k.startswith("sg_") or k in ("position", "total", "thru")
                    }

        # In-play probabilities
        in_play = results.get("in_play", {})
        ip_list = in_play.get("data", in_play.get("players", []))
        if isinstance(ip_list, list):
            for player_name in players:
                player = _find_player_in_list(player_name, ip_list)
                if player:
                    live.setdefault(player_name, {})["live_probs"] = {
                        "win": player.get("win_prob", player.get("win")),
                        "top_5": player.get("top_5_prob", player.get("top_5")),
                        "top_10": player.get("top_10_prob", player.get("top_10")),
                        "make_cut": player.get("make_cut_prob", player.get("make_cut")),
                    }

        return live

    async def close(self) -> None:
        """Close the underlying DataGolf client session."""
        await self._client.close()
