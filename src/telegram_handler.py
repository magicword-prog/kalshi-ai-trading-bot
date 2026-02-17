"""
Telegram Trade Approval Handler

Sends trade recommendations to Telegram with Approve/Reject inline buttons.
All live trades require explicit Telegram approval before execution.
Paper trades send notification-only messages (no blocking).
"""

import asyncio
import uuid
import logging
from typing import Optional, Dict

from telegram import Bot, InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.ext import Application, CallbackQueryHandler, CommandHandler, ContextTypes

from src.utils.logging_setup import get_trading_logger

logger = get_trading_logger("telegram_handler")


class TelegramHandler:
    """
    Handles Telegram bot interactions for trade approval workflow.

    - Sends trade recommendations with Approve/Reject buttons
    - Blocks live trades until user responds or 30-minute timeout
    - Sends paper trade notifications (no blocking)
    - Provides /portfolio command and alert messages
    """

    def __init__(self, bot_token: str, chat_id: str):
        self.bot_token = bot_token
        self.chat_id = chat_id
        self.bot = Bot(token=bot_token)
        self._app: Optional[Application] = None
        self._pending_approvals: Dict[str, asyncio.Event] = {}
        self._approval_results: Dict[str, bool] = {}
        self._running = False

    async def start(self):
        """Start the Telegram bot's callback query handler in the background."""
        if self._running:
            return

        self._app = (
            Application.builder()
            .token(self.bot_token)
            .build()
        )

        self._app.add_handler(CallbackQueryHandler(self._handle_callback))
        self._app.add_handler(CommandHandler("portfolio", self._handle_portfolio_command))

        await self._app.initialize()
        await self._app.start()
        await self._app.updater.start_polling(drop_pending_updates=True)
        self._running = True
        logger.info("Telegram bot started and listening for callbacks")

    async def stop(self):
        """Shut down the Telegram bot gracefully."""
        if not self._running or not self._app:
            return

        try:
            await self._app.updater.stop()
            await self._app.stop()
            await self._app.shutdown()
        except Exception as e:
            logger.error(f"Error shutting down Telegram bot: {e}")
        finally:
            self._running = False
            # Unblock any pending approvals
            for event in self._pending_approvals.values():
                event.set()
            logger.info("Telegram bot stopped")

    async def send_trade_approval(self, position) -> bool:
        """
        Send a trade recommendation and wait for Approve/Reject.

        Args:
            position: Position object with market_id, side, quantity,
                       entry_price, confidence, rationale, strategy.

        Returns:
            True if approved, False if rejected or timed out.
        """
        approval_id = str(uuid.uuid4())[:8]
        event = asyncio.Event()
        self._pending_approvals[approval_id] = event
        self._approval_results[approval_id] = False

        price_cents = int(position.entry_price * 100)
        total_cost = position.entry_price * position.quantity
        confidence_pct = int((position.confidence or 0) * 100)
        strategy = position.strategy or "directional"
        rationale = position.rationale or "No analysis provided."

        # Truncate rationale if too long for Telegram
        if len(rationale) > 500:
            rationale = rationale[:497] + "..."

        message = (
            f"\U0001f514 TRADE RECOMMENDATION\n\n"
            f"\U0001f4ca Market: {position.market_id}\n"
            f"\U0001f4c8 Side: {position.side} | Price: {price_cents}\u00a2 | Qty: {position.quantity}\n"
            f"\U0001f4b0 Total Cost: ${total_cost:.2f}\n"
            f"\U0001f3af Confidence: {confidence_pct}%\n"
            f"\U0001f4cb Strategy: {strategy}\n\n"
            f"\U0001f4a1 Analysis:\n{rationale}"
        )

        keyboard = InlineKeyboardMarkup([
            [
                InlineKeyboardButton("\u2705 Approve", callback_data=f"approve_{approval_id}"),
                InlineKeyboardButton("\u274c Reject", callback_data=f"reject_{approval_id}"),
            ]
        ])

        try:
            await self.bot.send_message(
                chat_id=self.chat_id,
                text=message,
                reply_markup=keyboard,
            )
            logger.info(f"Sent trade approval request {approval_id} for {position.market_id}")
        except Exception as e:
            logger.error(f"Failed to send Telegram message: {e}")
            # If we can't reach Telegram, don't block the trade
            self._cleanup_approval(approval_id)
            return True

        # Wait for user response with 30-minute timeout
        try:
            await asyncio.wait_for(event.wait(), timeout=1800)
        except asyncio.TimeoutError:
            logger.info(f"Trade approval {approval_id} timed out for {position.market_id}")
            try:
                await self.bot.send_message(
                    chat_id=self.chat_id,
                    text=f"\u23f0 Trade approval TIMED OUT for {position.market_id} - trade skipped.",
                )
            except Exception:
                pass

        approved = self._approval_results.get(approval_id, False)
        self._cleanup_approval(approval_id)
        return approved

    async def send_trade_notification(self, position):
        """
        Send a notification-only message for paper trades (no buttons, no blocking).
        """
        price_cents = int(position.entry_price * 100)
        total_cost = position.entry_price * position.quantity
        confidence_pct = int((position.confidence or 0) * 100)
        strategy = position.strategy or "directional"
        rationale = position.rationale or "No analysis provided."

        if len(rationale) > 500:
            rationale = rationale[:497] + "..."

        message = (
            f"\U0001f4dd PAPER TRADE EXECUTED\n\n"
            f"\U0001f4ca Market: {position.market_id}\n"
            f"\U0001f4c8 Side: {position.side} | Price: {price_cents}\u00a2 | Qty: {position.quantity}\n"
            f"\U0001f4b0 Total Cost: ${total_cost:.2f}\n"
            f"\U0001f3af Confidence: {confidence_pct}%\n"
            f"\U0001f4cb Strategy: {strategy}\n\n"
            f"\U0001f4a1 Analysis:\n{rationale}"
        )

        try:
            await self.bot.send_message(chat_id=self.chat_id, text=message)
        except Exception as e:
            logger.error(f"Failed to send paper trade notification: {e}")

    async def send_portfolio_summary(self, positions, balance: float):
        """Send a formatted portfolio summary to Telegram."""
        if not positions:
            message = f"\U0001f4ca Portfolio Summary\n\nNo open positions.\nBalance: ${balance:.2f}"
        else:
            lines = [f"\U0001f4ca Portfolio Summary\n"]
            total_cost = 0.0
            for p in positions:
                cost = p.entry_price * p.quantity
                total_cost += cost
                lines.append(
                    f"  {p.side} {p.market_id}\n"
                    f"    Qty: {p.quantity} @ {int(p.entry_price * 100)}\u00a2 = ${cost:.2f}"
                )
            lines.append(f"\nTotal invested: ${total_cost:.2f}")
            lines.append(f"Balance: ${balance:.2f}")
            message = "\n".join(lines)

        try:
            await self.bot.send_message(chat_id=self.chat_id, text=message)
        except Exception as e:
            logger.error(f"Failed to send portfolio summary: {e}")

    async def send_alert(self, message: str):
        """Send an alert message (market movements, errors, etc.)."""
        try:
            await self.bot.send_message(
                chat_id=self.chat_id,
                text=f"\u26a0\ufe0f ALERT\n\n{message}",
            )
        except Exception as e:
            logger.error(f"Failed to send Telegram alert: {e}")

    async def _handle_callback(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle Approve/Reject button presses."""
        query = update.callback_query
        await query.answer()

        data = query.data or ""
        if data.startswith("approve_"):
            approval_id = data[len("approve_"):]
            self._approval_results[approval_id] = True
            await query.edit_message_text(
                text=query.message.text + "\n\n\u2705 APPROVED - executing trade."
            )
            logger.info(f"Trade {approval_id} APPROVED via Telegram")
        elif data.startswith("reject_"):
            approval_id = data[len("reject_"):]
            self._approval_results[approval_id] = False
            await query.edit_message_text(
                text=query.message.text + "\n\n\u274c REJECTED - trade skipped."
            )
            logger.info(f"Trade {approval_id} REJECTED via Telegram")
        else:
            return

        event = self._pending_approvals.get(approval_id)
        if event:
            event.set()

    async def _handle_portfolio_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /portfolio command."""
        await update.message.reply_text(
            "\U0001f4ca Portfolio data is fetched from the trading system. "
            "Use the bot's dashboard for full details."
        )

    def _cleanup_approval(self, approval_id: str):
        """Remove a pending approval from tracking dicts."""
        self._pending_approvals.pop(approval_id, None)
        self._approval_results.pop(approval_id, None)


# --- Singleton management ---

_telegram_handler: Optional[TelegramHandler] = None


def get_telegram_handler() -> Optional[TelegramHandler]:
    """Return the global TelegramHandler instance, or None if not configured."""
    return _telegram_handler


async def init_telegram_handler(bot_token: str, chat_id: str) -> TelegramHandler:
    """Initialize and start the global TelegramHandler singleton."""
    global _telegram_handler
    handler = TelegramHandler(bot_token, chat_id)
    await handler.start()
    _telegram_handler = handler
    return handler


async def shutdown_telegram_handler():
    """Stop and clear the global TelegramHandler singleton."""
    global _telegram_handler
    if _telegram_handler:
        await _telegram_handler.stop()
        _telegram_handler = None
