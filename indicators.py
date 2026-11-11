# indicators.py
import numpy as np
import pandas as pd
import logging

def compute_atr(df, period=14):
    """
    Compute Average True Range (ATR).
    """
    try:
        h, l, c = df['high'].values, df['low'].values, df['close'].values
        tr = [max(h[i] - l[i], abs(h[i] - c[i-1]), abs(l[i] - c[i-1])) for i in range(1, len(df))]
        if len(tr) < period:
            period = len(tr)
        atr = np.mean(tr[-period:]) if tr else 0.0
        return atr
    except Exception as e:
        logging.error(f"Error computing ATR: {e}")
        return 0.0

def compute_vwap(df):
    """
    Compute Volume Weighted Average Price (VWAP).
    """
    try:
        vp = (df['close'] * df['tick_volume']).cumsum()
        v = df['tick_volume'].cumsum()
        vwap = (vp / v).iloc[-1]
        return vwap if np.isfinite(vwap) else df['close'].iloc[-1]
    except Exception as e:
        logging.error(f"Error computing VWAP: {e}")
        return df['close'].iloc[-1]

def compute_ema(df, span):
    """
    Compute Exponential Moving Average (EMA).
    """
    try:
        ema = df['close'].ewm(span=span, adjust=False).mean().iloc[-1]
        return ema if np.isfinite(ema) else df['close'].iloc[-1]
    except Exception as e:
        logging.error(f"Error computing EMA: {e}")
        return df['close'].iloc[-1]

def compute_ADX(df, period=14):
    """
    Compute Average Directional Index (ADX) manually.
    """
    try:
        if len(df) < period + 1:
            logging.warning(f"Insufficient data for ADX: {len(df)} < {period + 1}")
            return 25.0  # Neutral default
        h, l, c = df['high'], df['low'], df['close']
        # Calculate True Range and Directional Movement
        tr = []
        plus_dm = []
        minus_dm = []
        for i in range(1, len(df)):
            h_diff = h.iloc[i] - h.iloc[i-1]
            l_diff = l.iloc[i-1] - l.iloc[i]
            plus = h_diff if h_diff > l_diff and h_diff > 0 else 0
            minus = l_diff if l_diff > h_diff and l_diff > 0 else 0
            tr_val = max(h.iloc[i] - l.iloc[i], abs(h.iloc[i] - c.iloc[i-1]), abs(l.iloc[i] - c.iloc[i-1]))
            tr.append(tr_val)
            plus_dm.append(plus)
            minus_dm.append(minus)
        # Smooth TR, +DM, -DM
        def smooth(series, period):
            series = series[-(period + len(series) - 1):]
            smoothed = []
            for i in range(len(series)):
                if i < period - 1:
                    continue
                elif i == period - 1:
                    smoothed.append(sum(series[:period]) / period)
                else:
                    smoothed.append((smoothed[-1] * (period - 1) + series[i]) / period)
            return smoothed
        tr_smooth = smooth(tr, period)
        plus_dm_smooth = smooth(plus_dm, period)
        minus_dm_smooth = smooth(minus_dm, period)
        # Calculate +DI, -DI
        plus_di = [100 * p / t if t != 0 else 0 for p, t in zip(plus_dm_smooth, tr_smooth)]
        minus_di = [100 * m / t if t != 0 else 0 for m, t in zip(minus_dm_smooth, tr_smooth)]
        # Calculate DX and ADX
        dx = []
        for p, m in zip(plus_di, minus_di):
            diff = abs(p - m)
            sum_di = p + m
            dx.append(100 * diff / sum_di if sum_di != 0 else 0)
        # Smooth DX to get ADX
        adx = []
        for i in range(len(dx)):
            if i < period - 1:
                continue
            elif i == period - 1:
                adx.append(sum(dx[:period]) / period)
            else:
                adx.append((adx[-1] * (period - 1) + dx[i]) / period)
        return adx[-1] if adx else 25.0
    except Exception as e:
        logging.error(f"Error computing ADX: {e}")
        return 25.0

def compute_dynamic_params(df):
    """
    Compute dynamic parameters for trading decisions.
    """
    try:
        adx = compute_ADX(df, period=14)
        # Calculate RSI
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(window=14).mean().iloc[-1]
        avg_loss = loss.rolling(window=14).mean().iloc[-1]
        rs = avg_gain / avg_loss if avg_loss != 0 else 100.0
        rsi = 100 - (100 / (1 + rs)) if np.isfinite(rs) else 50.0
        # Dynamic regression factor based on ADX
        regression_factor = 1.0 + (adx / 100.0)  # Stronger trend -> wider channel
        # TP/SL multipliers based on volatility
        atr = compute_atr(df, period=14)
        volatility = df['close'].pct_change().std() * 100 if len(df) > 1 else 0.01
        multiplier_tp = 1.5 + min(volatility, 0.05)  # Cap at high volatility
        multiplier_sl = 2.0 + min(volatility, 0.05)
        return {
            'dynamic_regression_factor': regression_factor,
            'multiplier_tp': multiplier_tp,
            'multiplier_sl': multiplier_sl,
            'adx': adx,
            'rsi': rsi
        }
    except Exception as e:
        logging.error(f"Error computing dynamic params: {e}")
        return {
            'dynamic_regression_factor': 1.0,
            'multiplier_tp': 1.5,
            'multiplier_sl': 2.0,
            'adx': 25.0,
            'rsi': 50.0
        }