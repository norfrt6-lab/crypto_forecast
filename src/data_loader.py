"""
Data loading module.

Data source priority:
1. Local CSV cache (from a previous CCXT fetch)
2. CCXT Binance public endpoint (no API key required)
3. Bundled static CSV dataset (data/btc_usdt_daily.csv) — ships with the repo
   so the reviewer can always run the pipeline without network access.
"""

import logging
import time
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

# Bundled dataset that ships with the repository
BUNDLED_CSV = Path(__file__).resolve().parent.parent / "data" / "btc_usdt_daily.csv"


def _fetch_ccxt(
    symbol: str,
    timeframe: str,
    since_iso: str,
    max_candles: int = 3000,
) -> pd.DataFrame:
    """
    Fetch OHLCV data from Binance via CCXT (public, no API key).

    Paginates through results in batches of 1000 (Binance max per request).
    """
    import ccxt

    exchange = ccxt.binance({"enableRateLimit": True})
    since = exchange.parse8601(since_iso)
    limit = 1000
    all_candles = []

    while len(all_candles) < max_candles:
        logger.info(
            "CCXT: fetching from %s (%d candles so far)",
            exchange.iso8601(since),
            len(all_candles),
        )
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since=since, limit=limit)
        if not ohlcv:
            break
        all_candles.extend(ohlcv)
        since = ohlcv[-1][0] + 1  # +1ms to avoid duplicate last candle
        if len(ohlcv) < limit:
            break
        time.sleep(0.5)

    df = pd.DataFrame(
        all_candles, columns=["timestamp", "open", "high", "low", "close", "volume"]
    )
    df["date"] = pd.to_datetime(df["timestamp"], unit="ms")
    df = df.set_index("date")[["open", "high", "low", "close", "volume"]]
    df.index.name = "date"
    return df


def load_ohlcv(
    symbol: str = "BTC/USDT",
    interval: str = "1d",
    since: str = "2020-01-01T00:00:00Z",
    cache_path: str | None = None,
    static_csv: str | None = None,
) -> pd.DataFrame:
    """
    Load OHLCV data with a 3-tier fallback strategy.

    Returns a DataFrame indexed by datetime with columns:
        open, high, low, close, volume
    """
    # --- Tier 1: Local cache from previous run ---
    if cache_path:
        cache_dir = Path(cache_path)
        cache_dir.mkdir(parents=True, exist_ok=True)
        safe_symbol = symbol.replace("/", "_")
        cache_file = cache_dir / f"{safe_symbol}_{interval}.csv"
        if cache_file.exists():
            logger.info("Loading cached data from %s", cache_file)
            df = pd.read_csv(cache_file, index_col="date", parse_dates=True)
            logger.info(
                "Loaded %d rows (%s to %s)",
                len(df), df.index.min().date(), df.index.max().date(),
            )
            return df

    # --- Tier 2: CCXT Binance (public, no API key) ---
    try:
        logger.info("Fetching %s via CCXT (Binance public API)", symbol)
        df = _fetch_ccxt(symbol, interval, since)
        if df.empty:
            raise ValueError("CCXT returned no data")
        logger.info(
            "Downloaded %d rows (%s to %s)",
            len(df), df.index.min().date(), df.index.max().date(),
        )
        if cache_path:
            df.to_csv(cache_file)
            logger.info("Cached to %s", cache_file)
        return df

    except Exception as e:
        logger.warning("CCXT fetch failed: %s", e)

    # --- Tier 3: Bundled static CSV ---
    # Resolve relative paths against the project root (not cwd)
    if static_csv:
        csv_path = Path(static_csv)
        if not csv_path.is_absolute():
            csv_path = BUNDLED_CSV.parent.parent / csv_path
    else:
        csv_path = BUNDLED_CSV
    if csv_path.exists():
        logger.info("Loading bundled static dataset from %s", csv_path)
        df = pd.read_csv(csv_path, index_col="date", parse_dates=True)
        logger.info(
            "Loaded %d rows (%s to %s)",
            len(df), df.index.min().date(), df.index.max().date(),
        )
        return df

    raise RuntimeError(
        "All data sources failed. Ensure network access for CCXT, or place a "
        f"CSV at {csv_path}"
    )
