# config.py
import os
from dotenv import load_dotenv

# Force load .env with override
load_dotenv(override=True)

# MetaTrader 5 connection info
MT5_SERVER = os.getenv('MT5_SERVER', 'MetaQuotes-Demo')
# Credentials come only from the environment / a local .env file (see .env.example) - never hard-code them.
MT5_ACCOUNT_DEMO = int(os.getenv('MT5_ACCOUNT_DEMO', '0'))
MT5_PASSWORD_DEMO = os.getenv('MT5_PASSWORD_DEMO')
MT5_TERMINAL_PATH = os.getenv("MT5_TERMINAL_PATH", "C:\\Program Files\\MetaTrader 5\\terminal64.exe")

# Trading settings
SYMBOL = os.getenv('SYMBOL', 'EURUSD')
TIMEFRAME = os.getenv('TIMEFRAME', 'M5')
MIN_LOT_SIZE = float(os.getenv('MIN_LOT_SIZE', '0.01'))
MAX_LOT_SIZE = float(os.getenv('MAX_LOT_SIZE', '1.0'))
LOT_DECREMENT = float(os.getenv('LOT_DECREMENT', '0.01'))

# ML model settings
ML_MODEL_PATH = os.getenv('ML_MODEL_PATH', 'models/fakeout_model.pkl')
TRADE_THRESHOLD_FOR_RETRAIN = int(os.getenv('TRADE_THRESHOLD_FOR_RETRAIN', '100'))

# Risk management
# MAX_RISK here is the nominal risk per trade (in percentage terms of the account) as a reference.
# Its dollar equivalent will be computed dynamically.
MAX_RISK = float(os.getenv('MAX_RISK', '0.0025'))  # e.g. 0.25% risk per trade as a guideline.
MAX_TOTAL_RISK_PERCENT = float(os.getenv('MAX_TOTAL_RISK_PERCENT', '0.05'))  # Total risk cap (5% of account)

KELLY_WIN_PROBABILITY = float(os.getenv('KELLY_WIN_PROBABILITY', '0.55'))
KELLY_WIN_LOSS_RATIO = float(os.getenv('KELLY_WIN_LOSS_RATIO', '1.5'))
KELLY_MIN_FRACTION = float(os.getenv('KELLY_MIN_FRACTION', '0.05'))
KELLY_MAX_FRACTION = float(os.getenv('KELLY_MAX_FRACTION', '0.5'))
STOP_LOSS_PIPS = float(os.getenv('STOP_LOSS_PIPS', '6'))

# NEW: Expected adverse move in pips which defines the trade’s potential risk.
EXPECTED_HEDGE_PIPS = float(os.getenv('EXPECTED_HEDGE_PIPS', '10'))

# Scheduler settings (in minutes)
RETRAIN_INTERVAL = int(os.getenv('RETRAIN_INTERVAL', '60'))

# Trading parameters
SCALPING_WINDOW = int(os.getenv('SCALPING_WINDOW', '15'))
SMOOTHING_ALPHA = float(os.getenv('SMOOTHING_ALPHA', '0.7'))
FINAL_SIGNAL_THRESHOLD = float(os.getenv('FINAL_SIGNAL_THRESHOLD', '0.5'))
RAW_SIGNAL_ATR_FACTOR = float(os.getenv('RAW_SIGNAL_ATR_FACTOR', '0.4'))
TRAILING_TRIGGER = float(os.getenv('TRAILING_TRIGGER', '0.5'))
TRAILING_STOP_MULTIPLIER = float(os.getenv('TRAILING_STOP_MULTIPLIER', '0.5'))
HARD_STOP_MULTIPLIER = float(os.getenv('HARD_STOP_MULTIPLIER', '0.0'))
HEDGE_THRESHOLD = float(os.getenv('HEDGE_THRESHOLD', '0.02'))
ACCOUNT_LOSS_THRESHOLD = float(os.getenv('ACCOUNT_LOSS_THRESHOLD', '0.10'))
HEDGE_SIZE_RATIO = float(os.getenv('HEDGE_SIZE_RATIO', '0.3'))
SLIPPAGE_ATR_MULTIPLIER = float(os.getenv('SLIPPAGE_ATR_MULTIPLIER', '1.0'))
LOOP_SLEEP_TIME = int(os.getenv('LOOP_SLEEP_TIME', '5'))
DAILY_LOSS_THRESHOLD = float(os.getenv('DAILY_LOSS_THRESHOLD', '-500'))
COOL_DOWN_PERIOD = int(os.getenv('COOL_DOWN_PERIOD', '60'))
D1_SLOPE_THRESHOLD = float(os.getenv('D1_SLOPE_THRESHOLD', '0.1'))

# Mode selection: "0" for relaxed, "1" for strict
MODE = os.getenv('MODE', '0')

# Candidate forex pairs for trading
CANDIDATE_PAIRS = ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCHF", "NZDUSD", "USDCAD"]

# Trade history file
TRADE_HISTORY_FILE = os.getenv('TRADE_HISTORY_FILE', "trade_history.csv")
