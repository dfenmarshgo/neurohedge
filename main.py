# main.py
import os
import json
import numpy as np
import time
import logging
import csv
from datetime import datetime
import pandas as pd
from dotenv import load_dotenv
import MetaTrader5 as mt5
from stable_baselines3 import PPO

from risk_manager import RiskManager, load_active_trades, save_active_trades
import config
from mt5_connection import initialize, shutdown
from data_handler import scan_market
from ml_model import FakeoutClassifier
from mirror_engine import detect_liquidity_sweep
from shadow_sim import ShadowSimulator
from journal import JournalWriter
from pair_selector import select_best_pairs
from order_execution import send_live_order
from indicators import compute_dynamic_params, compute_vwap, compute_ema, compute_atr

load_dotenv(override=True)
USE_ML = False
weighted_signals = {pair: None for pair in config.CANDIDATE_PAIRS}
active_trades_cache = []

# Configure logging (INFO level for production; DEBUG messages are hidden)
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')
logging.info(f"Mode: {'strict' if config.MODE == '1' else 'relaxed'}")

def clear_active_trades_file():
    """Clear active trades file on startup so that stale trades do not count against total risk."""
    file_path = os.path.abspath("active_trades.json")
    try:
        with open(file_path, "w") as f:
            json.dump([], f)
        logging.info(f"Cleared active trades file: {file_path}")
    except Exception as e:
        logging.error(f"Error clearing active trades file: {e}")

def check_ml_toggle():
    global USE_ML
    trade_count = count_trades_in_csv()
    if trade_count >= config.TRADE_THRESHOLD_FOR_RETRAIN:
        USE_ML = True
        logging.info(f"Trade count {trade_count} reached {config.TRADE_THRESHOLD_FOR_RETRAIN}, enabling ML")
    else:
        logging.debug(f"Trade count {trade_count}/{config.TRADE_THRESHOLD_FOR_RETRAIN}, ML disabled")

def lt_regression_channel_advanced(prices, window=config.SCALPING_WINDOW, regression_factor=1.0):
    prices = np.array(prices)
    if len(prices) < window:
        window = max(3, len(prices))
    if len(prices) < 3:
        return 0.0, prices[-1], prices[-1], prices[-1]
    x = np.arange(window)
    y = prices[-window:]
    try:
        slope, intercept = np.polyfit(x, y, 1)
        fitted = intercept + slope * x
        mid = fitted[-1]
        residuals = y - fitted
        std = np.std(residuals, ddof=1) or np.std(y) / 2
        lower = mid - regression_factor * std
        upper = mid + regression_factor * std
        return slope, mid, lower, upper
    except Exception as e:
        logging.error(f"Regression channel error: {e}")
        return 0.0, prices[-1], prices[-1], prices[-1]

def get_multi_tf_slope(symbol, tf1, tf2, window=config.SCALPING_WINDOW, market_data_cache=None):
    if market_data_cache is None:
        market_data_cache = {}
    try:
        if (symbol, tf1) not in market_data_cache:
            market_data_cache[(symbol, tf1)] = scan_market(symbol=symbol, timeframe=tf1)
        df1 = market_data_cache[(symbol, tf1)]['data']
        s1, *_ = lt_regression_channel_advanced(df1['close'].values, window)
    except Exception:
        s1 = 0.0
    try:
        if (symbol, tf2) not in market_data_cache:
            market_data_cache[(symbol, tf2)] = scan_market(symbol=symbol, timeframe=tf2)
        df2 = market_data_cache[(symbol, tf2)]['data']
        s2, *_ = lt_regression_channel_advanced(df2['close'].values, window)
    except Exception:
        s2 = 0.0
    return s1, s2

def determine_trend_bias(slope_tf1, slope_tf2, threshold=0.00005):
    combined = (slope_tf1 + slope_tf2) / 2.0
    if combined > threshold:
        return "Bullish"
    elif combined < -threshold:
        return "Bearish"
    return "Neutral"

def get_long_term_trend(symbol, timeframe=mt5.TIMEFRAME_D1, period=50, market_data_cache=None):
    if market_data_cache is None:
        market_data_cache = {}
    try:
        if (symbol, timeframe) not in market_data_cache:
            market_data_cache[(symbol, timeframe)] = scan_market(symbol=symbol, timeframe=timeframe, num_bars=period+1)
        df = market_data_cache[(symbol, timeframe)]['data']
        if df.empty:
            return "Neutral"
        ema = compute_ema(df, period)
        prev_ema = compute_ema(df.iloc[:-1], period)
        slope = ema - prev_ema
        if slope > config.D1_SLOPE_THRESHOLD:
            return "Bullish"
        elif slope < -config.D1_SLOPE_THRESHOLD:
            return "Bearish"
        return "Neutral"
    except Exception:
        return "Neutral"

def append_trade_to_csv(data):
    exists = os.path.isfile(config.TRADE_HISTORY_FILE)
    hdr = ['timestamp', 'forex_pair', 'entry_price', 'exit_price', 'take_profit', 'stop_loss',
           'volatility', 'ATR', 'vwap_bias', 'multi_tf', 'liquidity', 'ML_confidence',
           'trade_size', 'trade_direction', 'outcome', 'method', 'rl_action']
    try:
        with open(config.TRADE_HISTORY_FILE, 'a', newline='') as f:
            w = csv.DictWriter(f, fieldnames=hdr)
            if not exists:
                w.writeheader()
            w.writerow(data)
    except Exception as e:
        logging.error(f"Error appending to trade CSV: {e}")

def count_trades_in_csv():
    file_path = os.path.abspath(config.TRADE_HISTORY_FILE)
    hdr = ['timestamp', 'forex_pair', 'entry_price', 'exit_price', 'take_profit', 'stop_loss',
           'volatility', 'ATR', 'vwap_bias', 'multi_tf', 'liquidity', 'ML_confidence',
           'trade_size', 'trade_direction', 'outcome', 'method', 'rl_action']
    if not os.path.isfile(file_path):
        try:
            with open(file_path, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=hdr)
                writer.writeheader()
            logging.info(f"Created empty {file_path}")
        except Exception as e:
            logging.error(f"Error creating trade CSV: {e}")
        return 0
    try:
        df = pd.read_csv(file_path)
        return len(df)
    except Exception:
        return 0

def get_session_risk_multiplier():
    current_time = datetime.now().time()
    high_liq_start = datetime.strptime("13:00", "%H:%M").time()
    high_liq_end = datetime.strptime("20:00", "%H:%M").time()
    return 0.8 if high_liq_start <= current_time <= high_liq_end else 0.5

def check_favorable_conditions(symbol, market_data_cache=None):
    if market_data_cache is None:
        market_data_cache = {}
    try:
        if (symbol, mt5.TIMEFRAME_D1) not in market_data_cache:
            market_data_cache[(symbol, mt5.TIMEFRAME_D1)] = scan_market(symbol=symbol, timeframe=mt5.TIMEFRAME_D1, num_bars=51)
        df = market_data_cache[(symbol, mt5.TIMEFRAME_D1)]['data']
        if df.empty:
            return False
        ema = compute_ema(df, 50)
        prev_ema = compute_ema(df.iloc[:-1], 50)
        slope = ema - prev_ema
        if (symbol, mt5.TIMEFRAME_H1) not in market_data_cache:
            market_data_cache[(symbol, mt5.TIMEFRAME_H1)] = scan_market(symbol=symbol, timeframe=mt5.TIMEFRAME_H1)
        h1_df = market_data_cache[(symbol, mt5.TIMEFRAME_H1)]['data']
        reg_slope, _, _, _ = lt_regression_channel_advanced(h1_df['close'].values)
        if slope > config.D1_SLOPE_THRESHOLD and reg_slope > 0:
            return True
        if slope < -config.D1_SLOPE_THRESHOLD and reg_slope < 0:
            return True
        return False
    except Exception:
        return False

def main():
    # Clear active trades at startup
    clear_active_trades_file()
    
    if not os.getenv('MT5_ACCOUNT_DEMO') or not os.getenv('MT5_PASSWORD_DEMO'):
        raise Exception("Set MT5_ACCOUNT_DEMO and MT5_PASSWORD_DEMO in a local .env file (see .env.example)")

    initialize()
    logging.info("MT5 initialized")

    for symbol in config.CANDIDATE_PAIRS:
        try:
            mt5.symbol_select(symbol, True)
            logging.info(f"Added {symbol} to Market Watch")
        except Exception:
            pass

    risk_manager = RiskManager(scalping_mode=True)
    shadow_sim = ShadowSimulator()
    journal = JournalWriter()
    model = FakeoutClassifier()
    model.load_model()
    try:
        rl_model = PPO.load("models/rl_agent.zip")
    except Exception:
        rl_model = None

    global active_trades_cache
    active_trades_cache = load_active_trades()
    daily_pnl = 0.0
    last_date = datetime.now().date()
    requote_count = 0
    paused = False

    while True:
        try:
            check_ml_toggle()
            today = datetime.now().date()
            if today != last_date:
                daily_pnl = 0.0
                last_date = today
                requote_count = 0
                logging.info("New day reset")

            if daily_pnl < config.DAILY_LOSS_THRESHOLD:
                logging.warning("Pausing due to daily loss")
                time.sleep(config.COOL_DOWN_PERIOD * 60)
                continue

            if not risk_manager.check_total_loss():
                paused = True
                logging.warning("Paused due to account loss threshold")
                active_trades_cache = []
                save_active_trades(active_trades_cache)
                risk_manager.total_risk = 0.0
                risk_manager.total_volume = 0.0

            if paused:
                market_data_cache = {}
                for pair in config.CANDIDATE_PAIRS:
                    if check_favorable_conditions(pair, market_data_cache):
                        paused = False
                        logging.info(f"Resuming: Favorable conditions for {pair}")
                        break
                if paused:
                    logging.info("Still paused: No favorable conditions")
                    time.sleep(config.COOL_DOWN_PERIOD * 60)
                    continue

            if requote_count >= 3:
                logging.warning("Too many requotes, pausing")
                time.sleep(30)
                requote_count = 0
                shutdown()
                initialize()
                continue

            market_data_cache = {}
            best_pairs = select_best_pairs(config.CANDIDATE_PAIRS)
            logging.info(f"Active trades: {len(active_trades_cache)}, total_risk: {risk_manager.total_risk:.2f}")
            for pair_info in best_pairs[:3]:
                pair = pair_info['symbol']
                if (pair, mt5.TIMEFRAME_M15) not in market_data_cache:
                    market_data_cache[(pair, mt5.TIMEFRAME_M15)] = scan_market(symbol=pair)
                data = market_data_cache[(pair, mt5.TIMEFRAME_M15)]
                df = data['data']
                if df.empty:
                    continue

                price = df['close'].iat[-1]
                atr = data['ATR']
                vol = data['volatility']
                vwap = compute_vwap(df)

                dynamic_params = compute_dynamic_params(df)
                reg_factor = dynamic_params['dynamic_regression_factor']
                adx = dynamic_params['adx']

                slope, mid, lower, upper = lt_regression_channel_advanced(df['close'].values, regression_factor=reg_factor)
                s5, s1 = get_multi_tf_slope(pair, mt5.TIMEFRAME_M5, mt5.TIMEFRAME_H1, market_data_cache=market_data_cache)
                trend_bias = determine_trend_bias(s5, s1)
                long_term_trend = get_long_term_trend(pair, market_data_cache=market_data_cache)
                fast_ema = compute_ema(df, span=5)
                slow_ema = compute_ema(df, span=20)
                ema_filter = "Bullish" if fast_ema > slow_ema else "Bearish"
                vwap_bias = "above" if price > vwap else "below"

                feat = [vol, data.get('trend', 0)] + [0] * 8
                ml_conf = model.predict([feat])[0] if model.fitted else 0.5

                rl_action = None
                if rl_model:
                    try:
                        if (pair, mt5.TIMEFRAME_D1) not in market_data_cache:
                            market_data_cache[(pair, mt5.TIMEFRAME_D1)] = scan_market(symbol=pair, timeframe=mt5.TIMEFRAME_D1, num_bars=51)
                        d1_data = market_data_cache[(pair, mt5.TIMEFRAME_D1)]
                        d1_slope = 0.0 if d1_data['data'].empty else compute_ema(d1_data['data'], 50) - compute_ema(d1_data['data'].iloc[:-1], 50)
                        obs = np.array([
                            price / df['close'].mean(),
                            atr / (data['avg_atr'] or atr or 1.0),
                            vol / (data['volatility'] or 1.0),
                            (price - vwap) / (atr or 1.0),
                            slope / (np.std(np.gradient(df['close'].values)) or 1.0),
                            (fast_ema - slow_ema) / (atr or 1.0),
                            s5 / (np.std(np.gradient(df['close'].values)) or 1.0),
                            s1 / (np.std(np.gradient(df['close'].values)) or 1.0),
                            df['tick_volume'].iloc[-1] / (df['tick_volume'].mean() or 1.0),
                            ml_conf,
                            1 if price > vwap else -1,
                            d1_slope / (np.std(np.gradient(d1_data['data']['close'].values)) or 1.0) if not d1_data['data'].empty else 0.0,
                            len(active_trades_cache)
                        ], dtype=np.float32)
                        rl_action, _ = rl_model.predict(obs, deterministic=True)
                        rl_action = int(rl_action)
                    except Exception:
                        pass

                liq = detect_liquidity_sweep({pair: df})
                up = liq.get(f"{pair}_up", False)
                dn = liq.get(f"{pair}_down", False)

                signal_factor = config.RAW_SIGNAL_ATR_FACTOR * (1.5 if adx < 20 else 1.0)
                dynamic_threshold = signal_factor * atr
                raw_signal = "long" if price <= lower + dynamic_threshold else "short" if price >= upper - dynamic_threshold else None

                numeric_signal = 1 if raw_signal == "long" else -1 if raw_signal == "short" else 0
                if weighted_signals[pair] is None:
                    weighted_signals[pair] = numeric_signal
                else:
                    weighted_signals[pair] = config.SMOOTHING_ALPHA * numeric_signal + (1 - config.SMOOTHING_ALPHA) * weighted_signals[pair]

                final_direction = None
                adaptive_signal_threshold = config.FINAL_SIGNAL_THRESHOLD * (1.2 if adx < 20 else 1.0)
                if weighted_signals[pair] >= adaptive_signal_threshold:
                    final_direction = "long"
                elif weighted_signals[pair] <= -adaptive_signal_threshold:
                    final_direction = "short"

                if (final_direction == "long" and long_term_trend == "Bearish") or \
                   (final_direction == "short" and long_term_trend == "Bullish"):
                    final_direction = None

                trade_method = "regression"
                if final_direction:
                    # Use a fixed trade size of 0.02 lots.
                    fixed_volume = 0.02
                    # Compute the dollar risk for this trade:
                    # risk = fixed_volume * (pip value for 1 lot) * EXPECTED_HEDGE_PIPS
                    pip_val_standard = risk_manager.get_symbol_pip_value(pair)
                    new_trade_risk_dollars = fixed_volume * pip_val_standard * config.EXPECTED_HEDGE_PIPS
                    # Check if adding this trade's risk stays below the total risk cap.
                    if not risk_manager.can_open_new_trade(new_trade_risk_dollars):
                        logging.info(f"Cannot open trade for {pair}: Total risk limit reached; no new trades.")
                        continue
                    entry = price
                    new_trade = {
                        'dir': final_direction,
                        'entry': entry,
                        'size': fixed_volume,
                        'max_fav': entry,
                        'min_fav': entry,
                        'timestamp': datetime.now().isoformat(),
                        'method': trade_method,
                        'symbol': pair,
                        'risk_percent': new_trade_risk_dollars / risk_manager.account_balance
                    }
                    active_trades_cache.append(new_trade)
                    save_active_trades(active_trades_cache)
                    ot = mt5.ORDER_TYPE_BUY if final_direction == "long" else mt5.ORDER_TYPE_SELL
                    res = send_live_order(pair, ot, fixed_volume, entry, 0.0, 0.0, atr=atr)
                    if res and res.retcode == mt5.TRADE_RETCODE_DONE:
                        requote_count = 0
                        journal.write_entry(f"{final_direction.upper()} {pair} @ {entry:.5f} size={fixed_volume:.2f}")
                        risk_manager.update_risk(new_trade_risk_dollars, add=True)
                        risk_manager.update_volume(fixed_volume, add=True)
                    else:
                        requote_count += 1
                        active_trades_cache.remove(new_trade)
                        save_active_trades(active_trades_cache)
                        logging.warning(f"Order failed for {pair}, requote_count={requote_count}")

            # Trade management for exiting and hedging (remains unchanged)
            for trade in active_trades_cache.copy():
                pair = trade['symbol']
                trade_method = trade.get('method', 'regression')
                if (pair, mt5.TIMEFRAME_M15) not in market_data_cache:
                    market_data_cache[(pair, mt5.TIMEFRAME_M15)] = scan_market(symbol=pair)
                data = market_data_cache[(pair, mt5.TIMEFRAME_M15)]
                df = data['data']
                if df.empty:
                    continue
                price = df['close'].iat[-1]
                atr = data['ATR']
                symbol_info = mt5.symbol_info(pair)
                pip_val_standard = risk_manager.get_symbol_pip_value(pair) if symbol_info else 0.1
                pips = (price - trade['entry']) * 10000 if trade['dir'] == "long" else (trade['entry'] - price) * 10000
                pnl = pips * trade['size'] * pip_val_standard
                if trade['dir'] == "long":
                    if price > trade['max_fav']:
                        trade['max_fav'] = price
                    trailing_trigger = trade['max_fav'] - price
                else:
                    if price < trade['min_fav']:
                        trade['min_fav'] = price
                    trailing_trigger = price - trade['min_fav']

                exit_flag = False
                trailing_stop = config.TRAILING_STOP_MULTIPLIER * atr
                min_profit = config.TRAILING_TRIGGER * atr
                loss_threshold = config.HEDGE_THRESHOLD * risk_manager.account_balance / trade['size'] / 10000
                if trade['dir'] == "long":
                    if trailing_trigger >= trailing_stop and (price - trade['entry']) >= min_profit:
                        exit_flag = True
                    elif price <= trade['entry'] - loss_threshold:
                        exit_flag = True
                else:
                    if trailing_trigger >= trailing_stop and (trade['entry'] - price) >= min_profit:
                        exit_flag = True
                    elif price >= trade['entry'] + loss_threshold:
                        exit_flag = True

                if trade['dir'] == "long" and price <= trade['entry'] - loss_threshold and 'hedge' not in trade:
                    hs = round(trade['size'] * config.HEDGE_SIZE_RATIO, 2)
                    hs = max(config.MIN_LOT_SIZE, hs)
                    trade['hedge'] = {'dir': 'short', 'entry': price, 'size': hs}
                    journal.write_entry(f"HEDGE SHORT {pair} @ {price:.5f} size={hs:.2f}")
                    shadow_sim.simulate_trade(price, None, None, "short")
                    res = send_live_order(pair, mt5.ORDER_TYPE_SELL, hs, price, 0.0, 0.0, atr=atr)
                    if res and res.retcode == mt5.TRADE_RETCODE_DONE:
                        risk_manager.update_volume(hs, add=True)
                elif trade['dir'] == "short" and price >= trade['entry'] + loss_threshold and 'hedge' not in trade:
                    hs = round(trade['size'] * config.HEDGE_SIZE_RATIO, 2)
                    hs = max(config.MIN_LOT_SIZE, hs)
                    trade['hedge'] = {'dir': 'long', 'entry': price, 'size': hs}
                    journal.write_entry(f"HEDGE LONG {pair} @ {price:.5f} size={hs:.2f}")
                    shadow_sim.simulate_trade(price, None, None, "long")
                    res = send_live_order(pair, mt5.ORDER_TYPE_BUY, hs, price, 0.0, 0.0, atr=atr)
                    if res and res.retcode == mt5.TRADE_RETCODE_DONE:
                        risk_manager.update_volume(hs, add=True)

                if exit_flag:
                    exit_price = price
                    daily_pnl += pnl
                    journal.write_entry(f"EXIT {trade['dir'].upper()} {pair} @ {exit_price:.5f} PnL={pnl:.2f}")
                    append_trade_to_csv({
                        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        'forex_pair': pair,
                        'entry_price': trade['entry'],
                        'exit_price': exit_price,
                        'take_profit': None,
                        'stop_loss': None,
                        'volatility': vol,
                        'ATR': atr,
                        'vwap_bias': price > vwap,
                        'multi_tf': trend_bias == "Bullish",
                        'liquidity': up if trade['dir'] == "long" else dn,
                        'ML_confidence': ml_conf,
                        'trade_size': trade['size'],
                        'trade_direction': trade['dir'],
                        'outcome': "win" if pnl > 0 else "loss",
                        'method': trade_method,
                        'rl_action': rl_action
                    })
                    active_trades_cache.remove(trade)
                    if 'hedge' in trade:
                        hedge_dir = "buy" if trade['hedge']['dir'] == "long" else "sell"
                        journal.write_entry(f"CLOSE HEDGE {hedge_dir.upper()} {pair} @ {price:.5f}")
                        shadow_sim.simulate_trade(price, None, None, hedge_dir)
                        hedge_ot = mt5.ORDER_TYPE_BUY if hedge_dir == "buy" else mt5.ORDER_TYPE_SELL
                        res = send_live_order(pair, hedge_ot, trade['hedge']['size'], price, 0.0, 0.0, atr=atr)
                        if res and res.retcode == mt5.TRADE_RETCODE_DONE:
                            risk_manager.update_volume(trade['hedge']['size'], add=False)
                    save_active_trades(active_trades_cache)
                    risk_manager.update_risk(trade['size'] * trade.get('risk_percent', config.MAX_RISK), add=False)
                    risk_manager.update_volume(trade['size'], add=False)

            time.sleep(config.LOOP_SLEEP_TIME)
        except KeyboardInterrupt:
            logging.info("Bot stopped by user")
            shutdown()
            break
        except Exception as e:
            logging.error(f"Main loop error: {e}", exc_info=True)
            shutdown()
            time.sleep(5)
            initialize()
            time.sleep(10)
    shutdown()

if __name__ == "__main__":
    main()
