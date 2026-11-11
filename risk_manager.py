# risk_manager.py
import logging
import MetaTrader5 as mt5
import os
import json
from datetime import datetime
import config

class RiskManager:
    def __init__(self, scalping_mode=True):
        self.account_balance = None
        self.total_risk = 0.0  # Total risk in dollars
        self.scalping_mode = scalping_mode
        self.total_volume = 0.0
        self.initialize_balance()

    def initialize_balance(self):
        try:
            if mt5.initialize():
                account_info = mt5.account_info()
                if account_info:
                    self.account_balance = account_info.balance
                    logging.info(f"RiskManager initialized with balance: {self.account_balance}")
                else:
                    logging.error("Failed to fetch account info")
                    self.account_balance = 1000.0
            else:
                logging.error("MT5 initialization failed")
                self.account_balance = 1000.0
        except Exception as e:
            logging.error(f"Error initializing balance: {e}")
            self.account_balance = 1000.0

    def get_symbol_pip_value(self, symbol):
        """Fetch pip value for the symbol (per 1 standard lot)."""
        try:
            symbol_info = mt5.symbol_info(symbol)
            if not symbol_info:
                logging.error(f"No symbol info for {symbol}")
                return 10.0  # Fallback (typical pip value for 1 standard lot)
            tick_size = symbol_info.point
            contract_size = symbol_info.trade_contract_size
            pip_value = contract_size * tick_size
            logging.debug(f"Pip value for {symbol}: {pip_value:.4f}")
            return pip_value
        except Exception as e:
            logging.error(f"Error getting pip value for {symbol}: {e}")
            return 10.0

    def can_open_new_trade(self, new_trade_risk_dollars):
        """
        Returns True if the cumulative risk (existing risk + new trade risk in dollars)
        is less than the maximum allowed risk in dollars (account_balance * MAX_TOTAL_RISK_PERCENT).
        """
        try:
            if self.account_balance <= 0:
                logging.error("Account balance is zero or negative")
                return False
            total_after = self.total_risk + new_trade_risk_dollars
            max_total_risk_dollars = config.MAX_TOTAL_RISK_PERCENT * self.account_balance
            logging.debug(f"Checking trade risk: current=${self.total_risk:.2f}, new=${new_trade_risk_dollars:.2f}, total=${total_after:.2f}, cap=${max_total_risk_dollars:.2f}")
            if total_after >= max_total_risk_dollars:
                logging.warning(f"Cumulative risk ${total_after:.2f} exceeds cap ${max_total_risk_dollars:.2f}")
                return False
            return True
        except Exception as e:
            logging.error(f"Error checking new trade risk: {e}")
            return False

    def update_risk(self, risk_dollars, add=True):
        try:
            if add:
                self.total_risk += risk_dollars
            else:
                self.total_risk = max(0.0, self.total_risk - risk_dollars)
            logging.debug(f"Updated total risk: ${self.total_risk:.2f}")
        except Exception as e:
            logging.error(f"Error updating risk: {e}")

    def update_volume(self, volume, add=True):
        try:
            if add:
                self.total_volume += volume
            else:
                self.total_volume = max(0.0, self.total_volume - volume)
            logging.debug(f"Updated total volume: {self.total_volume:.2f}")
        except Exception as e:
            logging.error(f"Error updating volume: {e}")

    def check_total_loss(self):
        try:
            trades = load_active_trades()
            total_pnl = 0.0
            for trade in trades:
                symbol = trade['symbol']
                entry = trade['entry']
                size = trade['size']
                direction = trade['dir']
                data = mt5.symbol_info_tick(symbol)
                if not data:
                    logging.warning(f"No tick data for {symbol}")
                    continue
                price = data.bid if direction == "long" else data.ask
                pip_value = self.get_symbol_pip_value(symbol)
                pips = (price - entry) * 10000 if direction == "long" else (entry - price) * 10000
                pnl = pips * size * pip_value
                total_pnl += pnl
            loss_threshold = config.ACCOUNT_LOSS_THRESHOLD * self.account_balance
            if total_pnl <= -loss_threshold:
                logging.warning(f"Total loss ${total_pnl:.2f} exceeds threshold -${loss_threshold:.2f}")
                return False
            return True
        except Exception as e:
            logging.error(f"Error checking total loss: {e}")
            return True

def load_active_trades():
    """
    Load active trades from active_trades.json and return valid trades (within 24 hours).
    """
    file_path = os.path.abspath("active_trades.json")
    try:
        if not os.path.isfile(file_path):
            with open(file_path, "w") as f:
                json.dump([], f)
            logging.info(f"Created {file_path} as []")
            return []
        with open(file_path, "r") as f:
            content = f.read().strip()
            logging.debug(f"Active trades file content: {content}")
            if not content:
                with open(file_path, "w") as fw:
                    json.dump([], fw)
                logging.info(f"Reset empty {file_path} to []")
                return []
            try:
                data = json.loads(content)
            except json.JSONDecodeError as e:
                logging.error(f"JSON decode error: {e}, content: '{content}'")
                with open(file_path, "w") as fw:
                    json.dump([], fw)
                logging.info(f"Reset invalid {file_path} to []")
                return []
            if not isinstance(data, list):
                logging.warning(f"Invalid active_trades.json format, resetting to []")
                with open(file_path, "w") as fw:
                    json.dump([], fw)
                return []
            valid_trades = []
            for trade in data:
                if ('symbol' in trade and trade['symbol'] in config.CANDIDATE_PAIRS and
                    'timestamp' in trade and 'dir' in trade and 'size' in trade):
                    try:
                        trade_time = datetime.fromisoformat(trade['timestamp'].replace('Z', '+00:00'))
                        if (datetime.now() - trade_time).total_seconds() < 24 * 3600:
                            valid_trades.append(trade)
                    except ValueError:
                        continue
            logging.info(f"Loaded {len(valid_trades)} valid active trades")
            return valid_trades
    except Exception as e:
        logging.error(f"Error loading active trades: {e}")
        with open(file_path, "w") as fw:
            json.dump([], fw)
        logging.info(f"Reset {file_path} to []")
        return []

def save_active_trades(trades):
    """
    Save the given trades to active_trades.json.
    """
    file_path = os.path.abspath("active_trades.json")
    try:
        with open(file_path, "w") as f:
            json.dump(trades, f, default=str)
        logging.debug(f"Saved {len(trades)} active trades to {file_path}")
    except Exception as e:
        logging.error(f"Error saving active trades: {e}")
