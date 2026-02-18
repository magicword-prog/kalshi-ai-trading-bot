"""
DataGolf API async client.

Fetches player stats, model predictions, field updates, and betting odds
from the DataGolf API (https://feeds.datagolf.com).

Auth: API key appended as query parameter ``key=<API_KEY>``.
"""

import asyncio
from typing import Any, Dict, List, Optional

import aiohttp

from src.config.settings import settings
from src.utils.logging_setup import get_trading_logger

logger = get_trading_logger("datagolf_client")

BASE_URL = "https://feeds.datagolf.com"


class DataGolfClient:
    """Async client for the DataGolf API."""

    def __init__(self, api_key: Optional[str] = None, tour: Optional[str] = None):
        self._api_key = api_key or settings.golf.datagolf_api_key
        self._tour = tour or settings.golf.tour
        self._session: Optional[aiohttp.ClientSession] = None

    # ------------------------------------------------------------------
    # Session management
    # ------------------------------------------------------------------
    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=30)
            )
        return self._session

    async def close(self) -> None:
        if self._session and not self._session.closed:
            await self._session.close()

    # ------------------------------------------------------------------
    # Generic request helper
    # ------------------------------------------------------------------
    async def _get(
        self, path: str, params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Make an authenticated GET request to DataGolf.

        Returns parsed JSON response or an empty dict on failure.
        """
        session = await self._get_session()
        query: Dict[str, Any] = {"key": self._api_key}
        if params:
            query.update(params)

        url = f"{BASE_URL}{path}"
        try:
            async with session.get(url, params=query) as resp:
                if resp.status != 200:
                    body = await resp.text()
                    logger.warning(
                        "DataGolf API error",
                        path=path,
                        status=resp.status,
                        body=body[:300],
                    )
                    return {}
                data = await resp.json()
                logger.debug("DataGolf response", path=path, keys=list(data.keys()) if isinstance(data, dict) else "list")
                return data if isinstance(data, dict) else {"data": data}
        except Exception as exc:
            logger.error("DataGolf request failed", path=path, error=str(exc))
            return {}

    # ------------------------------------------------------------------
    # Pre-Tournament Predictions
    # ------------------------------------------------------------------
    async def get_pre_tournament_predictions(
        self,
        tour: Optional[str] = None,
        odds_format: str = "percent",
    ) -> Dict[str, Any]:
        """
        Pre-tournament win/top-5/top-10/top-20/make-cut probabilities.

        Endpoint: /preds/pre-tournament
        """
        return await self._get(
            "/preds/pre-tournament",
            params={
                "tour": tour or self._tour,
                "odds_format": odds_format,
            },
        )

    # ------------------------------------------------------------------
    # Player Skill Decompositions
    # ------------------------------------------------------------------
    async def get_player_decompositions(
        self, tour: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        SG breakdown (OTT, APP, ARG, PUTT) per player for the upcoming event.

        Endpoint: /preds/player-decompositions
        """
        return await self._get(
            "/preds/player-decompositions",
            params={"tour": tour or self._tour},
        )

    # ------------------------------------------------------------------
    # Skill Ratings
    # ------------------------------------------------------------------
    async def get_skill_ratings(
        self, tour: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Overall skill estimates and rankings.

        Endpoint: /preds/skill-ratings
        """
        return await self._get(
            "/preds/skill-ratings",
            params={"tour": tour or self._tour},
        )

    # ------------------------------------------------------------------
    # Field Updates
    # ------------------------------------------------------------------
    async def get_field_updates(
        self, tour: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Current field, WDs, tee times.

        Endpoint: /field-updates
        """
        return await self._get(
            "/field-updates",
            params={"tour": tour or self._tour},
        )

    # ------------------------------------------------------------------
    # Live In-Play Predictions
    # ------------------------------------------------------------------
    async def get_in_play_predictions(
        self, tour: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Live finish probabilities during a tournament.

        Endpoint: /preds/in-play
        """
        return await self._get(
            "/preds/in-play",
            params={"tour": tour or self._tour},
        )

    # ------------------------------------------------------------------
    # Live Tournament Stats
    # ------------------------------------------------------------------
    async def get_live_tournament_stats(
        self,
        tour: Optional[str] = None,
        stats: str = "sg_putt,sg_arg,sg_app,sg_ott,sg_t2g,sg_total",
    ) -> Dict[str, Any]:
        """
        Live SG stats per player per round.

        Endpoint: /preds/live-tournament-stats
        """
        return await self._get(
            "/preds/live-tournament-stats",
            params={
                "tour": tour or self._tour,
                "stats": stats,
            },
        )

    # ------------------------------------------------------------------
    # Outright Odds (Betting Tools)
    # ------------------------------------------------------------------
    async def get_outright_odds(
        self,
        tour: Optional[str] = None,
        market: str = "win",
        odds_format: str = "percent",
    ) -> Dict[str, Any]:
        """
        DG model odds vs sportsbook odds for outright markets.

        Endpoint: /betting-tools/outrights
        market: "win", "top_5", "top_10", "top_20", "make_cut"
        """
        return await self._get(
            "/betting-tools/outrights",
            params={
                "tour": tour or self._tour,
                "market": market,
                "odds_format": odds_format,
            },
        )

    # ------------------------------------------------------------------
    # Matchup Odds (Betting Tools)
    # ------------------------------------------------------------------
    async def get_matchup_odds(
        self,
        tour: Optional[str] = None,
        odds_format: str = "percent",
    ) -> Dict[str, Any]:
        """
        Head-to-head matchup predictions.

        Endpoint: /betting-tools/matchups
        """
        return await self._get(
            "/betting-tools/matchups",
            params={
                "tour": tour or self._tour,
                "odds_format": odds_format,
            },
        )
