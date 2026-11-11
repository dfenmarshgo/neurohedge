# pair_selector.py
import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import logging
from data_handler import scan_market
from indicators import compute_atr
import config

def select_pair():
    """
    Select a single forex pair.
    """
    try:
        pairs = config.CANDIDATE_PAIRS
        if not pairs:
            logging.warning("No candidate pairs, defaulting to EURUSD")
            return "EURUSD"

        scores = []
        for pair in pairs:
            try:
                data = scan_market(symbol=pair, timeframe=mt5.TIMEFRAME_H1, num_bars=100)
                df = data['data']
                if df.empty:
                    continue
                atr = compute_atr(df, period=14)
                vol = df['close'].pct_change().rolling(20).std().iloc[-1]
                ema_fast = df['close'].ewm(span=5).mean().iloc[-1]
                ema_slow = df['close'].ewm(span=20).mean().iloc[-1]
                trend_score = abs(ema_fast - ema_slow) / atr if atr != 0 else 0
                score = vol + trend_score
                scores.append({"symbol": pair, "score": score})
            except Exception as e:
                logging.error(f"Error evaluating {pair}: {e}")
                continue

        if not scores:
            logging.warning("No valid pairs, defaulting to EURUSD")
            return "EURUSD"

        selected_pair = max(scores, key=lambda x: x["score"])["symbol"]
        logging.info(f"Selected pair: {selected_pair}")
        return selected_pair
    except Exception as e:
        logging.error(f"Pair selection failed: {e}")
        return "EURUSD"

def select_best_pairs(pairs, n=3):
    """
    Select top n forex pairs from provided list.
    """
    try:
        if not pairs:
            logging.warning("No candidate pairs, defaulting to [EURUSD]")
            return [{"symbol": "EURUSD", "score": 0.0}]

        scores = []
        for pair in pairs:
            try:
                data = scan_market(symbol=pair, timeframe=mt5.TIMEFRAME_H1, num_bars=100)
                df = data['data']
                if df.empty:
                    continue
                atr = compute_atr(df, period=14)
                vol = df['close'].pct_change().rolling(20).std().iloc[-1]
                ema_fast = df['close'].ewm(span=5).mean().iloc[-1]
                ema_slow = df['close'].ewm(span=20).mean().iloc[-1]
                trend_score = abs(ema_fast - ema_slow) / atr if atr != 0 else 0
                score = vol + trend_score
                scores.append({"symbol": pair, "score": score})
            except Exception as e:
                logging.error(f"Error evaluating {pair}: {e}")
                continue

        if not scores:
            logging.warning("No valid pairs, defaulting to [EURUSD]")
            return [{"symbol": "EURUSD", "score": 0.0}]

        scores.sort(key=lambda x: x["score"], reverse=True)
        selected_pairs = scores[:n]
        logging.info(f"Selected pairs: {[p['symbol'] for p in selected_pairs]}")
        return selected_pairs
    except Exception as e:
        logging.error(f"Pair selection failed: {e}")
        return [{"symbol": "EURUSD", "score": 0.0}]