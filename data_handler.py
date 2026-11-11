# data_handler.py
import pandas as pd
import numpy as np
import MetaTrader5 as mt5
import logging
from indicators import compute_atr

def scan_market(symbol='EURUSD', timeframe=mt5.TIMEFRAME_M5, num_bars=1000):
    """
    Fetch market data for a given symbol and timeframe.
    """
    try:
        # Ensure MT5 is initialized
        if not mt5.initialize():
            logging.error("MT5 initialization failed in scan_market")
            return {'data': pd.DataFrame(), 'ATR': 0.0, 'volatility': 0.0}

        # Select symbol
        if not mt5.symbol_select(symbol, True):
            logging.error(f"Failed to select symbol: {symbol}")
            return {'data': pd.DataFrame(), 'ATR': 0.0, 'volatility': 0.0}

        # Verify symbol exists
        symbol_info = mt5.symbol_info(symbol)
        if symbol_info is None or not symbol_info.visible:
            logging.error(f"Symbol {symbol} not available or not visible")
            return {'data': pd.DataFrame(), 'ATR': 0.0, 'volatility': 0.0}

        # Fetch OHLCV data
        rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, num_bars)
        if rates is None or len(rates) == 0:
            logging.error(f"No data returned for {symbol} on {timeframe}")
            return {'data': pd.DataFrame(), 'ATR': 0.0, 'volatility': 0.0}

        # Convert to DataFrame
        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        df = df[['time', 'open', 'high', 'low', 'close', 'tick_volume']]
        df.set_index('time', inplace=True)

        # Calculate ATR
        atr = compute_atr(df, period=14)

        # Calculate volatility (std of returns)
        returns = df['close'].pct_change().dropna()
        volatility = returns.std() if len(returns) > 0 else 0.0

        # Calculate average ATR over longer period
        avg_atr = compute_atr(df, period=50) if len(df) >= 50 else atr

        logging.debug(f"Loaded market data for {symbol}, shape: {df.shape}")
        return {
            'data': df,
            'ATR': atr,
            'volatility': volatility,
            'avg_atr': avg_atr
        }

    except Exception as e:
        logging.error(f"Data fetch failed for {symbol}: {e}")
        return {'data': pd.DataFrame(), 'ATR': 0.0, 'volatility': 0.0}
    finally:
        # Avoid shutting down MT5 to maintain connection
        pass

def get_available_symbols():
    """
    Retrieve all available symbols from MT5.
    """
    try:
        if not mt5.initialize():
            logging.error("MT5 initialization failed in get_available_symbols")
            return []
        symbols = mt5.symbols_get()
        if not symbols:
            logging.error("No symbols retrieved")
            return []
        return [s.name for s in symbols if s.visible]
    except Exception as e:
        logging.error(f"Failed to get symbols: {e}")
        return []
    finally:
        pass
