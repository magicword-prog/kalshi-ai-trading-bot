#!/usr/bin/env python3
"""Quick test: skip ingestion (markets already in DB), run one trading cycle with Telegram."""

import asyncio
from dotenv import load_dotenv
load_dotenv()

async def main():
    from src.utils.logging_setup import setup_logging, get_trading_logger
    from src.utils.database import DatabaseManager
    from src.clients.kalshi_client import KalshiClient
    from src.config.settings import settings
    from src.telegram_handler import init_telegram_handler, shutdown_telegram_handler, get_telegram_handler
    from src.jobs.trade import run_trading_job

    setup_logging(log_level='INFO')
    logger = get_trading_logger('test_pipeline')

    settings.trading.live_trading_enabled = False
    settings.trading.paper_trading_mode = True

    # Init DB (tables already exist from previous run)
    db_manager = DatabaseManager()
    await db_manager.initialize()

    # Check how many markets we have
    import aiosqlite
    async with aiosqlite.connect(db_manager.db_path) as db:
        cursor = await db.execute("SELECT COUNT(*) FROM markets")
        count = (await cursor.fetchone())[0]
    logger.info(f"Markets in DB: {count}")

    # Init Telegram
    if settings.api.telegram_bot_token and settings.api.telegram_chat_id:
        tg = await init_telegram_handler(settings.api.telegram_bot_token, settings.api.telegram_chat_id)
        logger.info("Telegram handler started")
        await tg.send_alert(f"Pipeline test starting. {count} markets in DB. Paper trading mode.")
    else:
        logger.warning("No Telegram credentials - skipping")
        tg = None

    # Run one trading cycle (this calls the full AI analysis pipeline)
    logger.info("Running one trading cycle...")
    try:
        results = await run_trading_job()
        if results and results.total_positions > 0:
            msg = f"Cycle complete - {results.total_positions} positions, ${results.total_capital_used:.2f} capital used"
            logger.info(msg)
            if tg:
                await tg.send_alert(msg)
        else:
            logger.info("Cycle complete - no positions created")
            if tg:
                await tg.send_alert("Trading cycle complete - no positions created this round.")
    except Exception as e:
        logger.error(f"Trading cycle error: {e}")
        if tg:
            await tg.send_alert(f"Trading cycle error: {e}")

    # Cleanup
    await shutdown_telegram_handler()
    logger.info("Test done")

asyncio.run(main())
