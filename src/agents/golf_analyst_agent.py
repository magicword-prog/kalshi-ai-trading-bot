"""
Golf Analyst Agent — expert PGA Tour analyst for the ensemble debate.

Reads ``market_data["golf_context"]`` (produced by GolfDataFetcher) and
provides deep golf-specific analysis using Strokes Gained data, course fit,
player form, and DataGolf model predictions.
"""

import json
from typing import Any, Dict, List

from src.agents.base_agent import BaseAgent


class GolfAnalystAgent(BaseAgent):
    """Golf-specialist agent for the ensemble debate pipeline."""

    AGENT_NAME = "golf_analyst"
    AGENT_ROLE = "golf_analyst"
    DEFAULT_MODEL = "claude-sonnet-4-5-20250929"

    SYSTEM_PROMPT = (
        "You are an expert PGA Tour analyst specializing in prediction markets. "
        "You have deep knowledge of Strokes Gained analytics, course fit, player "
        "form cycles, and how these translate to tournament outcomes.\n\n"
        "When analyzing markets, you consider:\n"
        "- Strokes Gained decomposition (OTT, APP, ARG, PUTT) and how it fits the course\n"
        "- Recent form (last 3-5 events) vs longer-term skill baseline\n"
        "- Course history and course-type fit (links, parkland, desert, etc.)\n"
        "- Field strength and how it affects top-N probabilities\n"
        "- Weather conditions and how they favor certain player profiles\n"
        "- DataGolf model predictions as a baseline, then adjust with your analysis\n\n"
        "You are looking for MISPRICINGS — where the market price differs from true "
        "probability based on golf-specific factors the market may be ignoring.\n\n"
        "Return your analysis as a JSON object (inside a ```json``` code block) "
        "with the following keys:\n"
        '  "probability": float (0.0-1.0, your estimated TRUE probability for this outcome),\n'
        '  "confidence": float (0.0-1.0, how confident you are in your estimate),\n'
        '  "dg_model_prob": float or null (DataGolf model probability if available),\n'
        '  "edge_vs_market": float (-1.0 to 1.0, your prob minus market implied prob),\n'
        '  "key_factors": list of strings (2-5 most important factors),\n'
        '  "sg_analysis": string (how SG profile fits this course/market),\n'
        '  "reasoning": string (detailed analysis and mispricing thesis)'
    )

    def _build_prompt(self, market_data: dict, context: dict) -> str:
        summary = self.format_market_summary(market_data)
        golf = market_data.get("golf_context", {})

        sections = [
            f"Analyse the following PGA golf prediction market.\n\n{summary}",
        ]

        if golf:
            sections.append(self._format_golf_context(golf, market_data))
        else:
            sections.append(
                "\n[No DataGolf analytics available. Use your general golf knowledge.]"
            )

        # Include other agents' results if available
        if context.get("forecaster_result"):
            fr = context["forecaster_result"]
            sections.append(
                f"\n--- FORECASTER ESTIMATE ---\n"
                f"Probability: {fr.get('probability', 'N/A')}\n"
                f"Reasoning: {str(fr.get('reasoning', ''))[:300]}"
            )

        sections.append(
            "\nUsing the DataGolf analytics above as your statistical foundation, "
            "estimate the TRUE probability for this market outcome and identify "
            "any mispricing.\n"
            "Return ONLY a JSON object inside a ```json``` code block."
        )

        return "\n".join(sections)

    def _format_golf_context(self, golf: dict, market_data: dict) -> str:
        """Format golf_context dict into a structured briefing for the LLM."""
        lines = ["\n--- DATAGOLF ANALYTICS ---"]

        # Market parsing info
        parsed = golf.get("market_parsed", {})
        if parsed:
            lines.append(f"Market Type: {parsed.get('market_type', 'unknown')}")
            lines.append(f"Players: {', '.join(parsed.get('players', []))}")
            if parsed.get("tournament"):
                lines.append(f"Tournament: {parsed['tournament']}")
            if parsed.get("top_n"):
                lines.append(f"Top-N Cutoff: {parsed['top_n']}")

        # Tournament info
        tourney = golf.get("tournament_info", {})
        if tourney:
            lines.append(f"\n-- Tournament Info --")
            for k, v in tourney.items():
                lines.append(f"  {k}: {v}")

        # Player stats
        player_stats = golf.get("player_stats", {})
        for player_name, stats in player_stats.items():
            lines.append(f"\n-- Player: {player_name} --")
            sg = stats.get("sg_breakdown", {})
            if sg:
                lines.append("  Strokes Gained:")
                for metric, val in sg.items():
                    if val is not None:
                        lines.append(f"    {metric}: {val}")
            skill = stats.get("skill_rating", {})
            if skill:
                lines.append(f"  Skill Rating: {skill.get('skill_estimate', 'N/A')}")
                lines.append(f"  Ranking: {skill.get('ranking', 'N/A')}")

        # Model predictions
        preds = golf.get("model_predictions", {})
        if preds:
            lines.append(f"\n-- DataGolf Model Predictions --")
            for player_name, probs in preds.items():
                lines.append(f"  {player_name}:")
                for outcome, prob in probs.items():
                    if prob is not None:
                        lines.append(f"    {outcome}: {prob}")

        # DG odds / edge signals
        dg_odds = golf.get("dg_odds", {})
        if dg_odds:
            lines.append(f"\n-- DataGolf Odds (Edge Signal) --")
            for player_name, odds in dg_odds.items():
                lines.append(f"  {player_name}: {odds}")

        # Matchup data
        matchup = golf.get("matchup_data", {})
        if matchup:
            lines.append(f"\n-- Matchup Data --")
            for k, v in matchup.items():
                lines.append(f"  {k}: {v}")

        # Live data
        live = golf.get("live_data", {})
        if live:
            lines.append(f"\n-- Live Tournament Data --")
            for player_name, ldata in live.items():
                lines.append(f"  {player_name}: {json.dumps(ldata, default=str)[:500]}")

        # Market price for edge comparison
        yes_price = market_data.get("yes_price")
        if yes_price is not None:
            lines.append(f"\nKalshi Market YES Price: {yes_price * 100:.1f}c (implied prob: {yes_price:.1%})")

        lines.append("--- END DATAGOLF ANALYTICS ---")
        return "\n".join(lines)

    def _parse_result(self, raw_json: dict) -> dict:
        probability = self.clamp(raw_json.get("probability", 0.5))
        confidence = self.clamp(raw_json.get("confidence", 0.5))
        dg_model_prob = raw_json.get("dg_model_prob")
        if dg_model_prob is not None:
            dg_model_prob = self.clamp(float(dg_model_prob))
        edge_vs_market = self.clamp(
            raw_json.get("edge_vs_market", 0.0), lo=-1.0, hi=1.0
        )
        key_factors = raw_json.get("key_factors", [])
        if not isinstance(key_factors, list):
            key_factors = [str(key_factors)]
        key_factors = [str(f) for f in key_factors][:10]

        sg_analysis = str(raw_json.get("sg_analysis", ""))
        reasoning = str(raw_json.get("reasoning", "No reasoning provided."))

        return {
            "probability": probability,
            "confidence": confidence,
            "dg_model_prob": dg_model_prob,
            "edge_vs_market": edge_vs_market,
            "key_factors": key_factors,
            "sg_analysis": sg_analysis,
            "reasoning": reasoning,
        }
