# mirror_engine.py
import pandas as pd

def detect_liquidity_sweep(data_dict):
    """
    Detect liquidity sweeps across correlated pairs.
    
    Parameters:
        data_dict (dict): Keys are symbols, and values are their respective data frames.
    
    Returns:
        dict: A dictionary with symbols as keys and boolean flags as values.
    """
    sweep_signals = {}
    for symbol, df in data_dict.items():
        # For example: if the absolute percentage change in close price spikes above 0.5%
        if df['close'].pct_change().abs().max() > 0.005:
            sweep_signals[symbol] = True
        else:
            sweep_signals[symbol] = False
    return sweep_signals
