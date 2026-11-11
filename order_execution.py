# order_execution.py
import MetaTrader5 as mt5
import logging
import time
import config

def send_live_order(symbol, order_type, volume, price, sl=0.0, tp=0.0, atr=0.0, deviation=10, retries=3):
    """
    Send a live trade order to MT5 with retries.
    """
    attempt = 0
    while attempt < retries:
        try:
            if not mt5.initialize():
                logging.error("MT5 initialization failed in send_live_order")
                return None

            symbol_info = mt5.symbol_info(symbol)
            if symbol_info is None or not symbol_info.visible:
                logging.error(f"Symbol {symbol} not available or not visible")
                mt5.symbol_select(symbol, True)
                time.sleep(1)
                symbol_info = mt5.symbol_info(symbol)
                if symbol_info is None:
                    return None

            point = symbol_info.point
            digits = symbol_info.digits

            # Get fresh tick price
            tick = mt5.symbol_info_tick(symbol)
            if tick is None:
                logging.error(f"Failed to get tick data for {symbol}")
                return None
            current_price = tick.ask if order_type == mt5.ORDER_TYPE_BUY else tick.bid
            price = round(current_price, digits)  # Use tick price
            if sl > 0:
                sl = round(sl, digits)
            if tp > 0:
                tp = round(tp, digits)

            slippage = abs(current_price - price)
            max_slippage = config.SLIPPAGE_ATR_MULTIPLIER * atr if atr > 0 else 0.0005
            if slippage > max_slippage:
                logging.warning(f"Price slippage too high: {slippage:.5f} > {max_slippage:.5f}")
                return None

            if volume < config.MIN_LOT_SIZE:
                logging.error(f"Invalid volume: {volume}, minimum is {config.MIN_LOT_SIZE}")
                return None

            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": symbol,
                "volume": volume,
                "type": order_type,
                "price": price,
                "sl": sl,
                "tp": tp,
                "deviation": deviation,
                "magic": 234000,
                "comment": "NeuroHedge live trade",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_IOC,
            }

            result = mt5.order_send(request)
            if result is None or result.retcode != mt5.TRADE_RETCODE_DONE:
                logging.error(f"Order failed: {result}")
                if result and result.retcode == mt5.TRADE_RETCODE_NO_PRICES:
                    logging.warning(f"Retrying due to no prices, attempt {attempt + 1}/{retries}")
                    attempt += 1
                    time.sleep(1)
                    continue
                return result
            logging.info(f"Order successful: {result}")
            return result
        except Exception as e:
            logging.error(f"Error sending order: {e}")
            attempt += 1
            time.sleep(1)
        finally:
            pass
    logging.error(f"Failed to send order after {retries} attempts")
    return None