# NeuroHedge

An experimental forex scalping bot for MetaTrader 5 that combines classic technical-indicator confluence with machine learning: a gradient-boosting "fakeout" filter that retrains on the bot's own trade history, and a reinforcement-learning (PPO) agent trained in a custom Gymnasium environment. It has layered risk controls, including exposure caps, daily loss limits and adverse-move hedging.

> **Personal research project. Demo accounts only.** This is not financial advice and not a production trading system. It has not been backtested to a standard that would justify live capital.

**Stack:** Python · pandas · NumPy · scikit-learn · Gymnasium · Stable-Baselines3 (PPO) · MetaTrader5 API · python-dotenv · schedule

---

## What it does

1. **Market data and indicators** (`data_handler.py`, `indicators.py`): pulls bars from MT5 and computes a volatility-adjusted regression channel, ATR, VWAP, EMAs, RSI, ADX and MACD.
2. **Signal generation** (`main.py`): raw long/short signals come from the regression channel. They are confirmed by momentum, trend and multi-timeframe checks, then smoothed with an EWMA so a trade fires only when the weighted signal clears a threshold.
3. **ML filter** (`ml_model.py`): a `GradientBoostingClassifier` scores whether a setup is likely a "fakeout". It is enabled once enough trades have been logged, and it retrains from `trade_history.csv` on a schedule (`scheduler.py`).
4. **Reinforcement learning** (`rl_environment.py`, `train_rl.py`): a custom Gymnasium environment (`NeuroHedgeEnv`) and PPO training via Stable-Baselines3. A trained agent (`models/rl_agent.zip`) can be loaded by the main loop.
5. **Risk management** (`risk_manager.py`, `main.py`):
   - ATR-based stop loss and take profit, with multipliers adjusted by ADX and RSI.
   - A cap on total open risk, as a percentage of the account.
   - An account-level loss threshold.
   - A daily loss limit with a cool-down period.
   - Time-of-day (session) risk multipliers.
6. **Execution and hedging** (`order_execution.py`, `main.py`): market orders with deviation handling. If a position moves against the entry by more than a threshold, a hedge at a configurable fraction of the position (default 30%) is opened.
7. **Journal and shadow simulation** (`journal.py`, `shadow_sim.py`, `mirror_engine.py`): trade logging and a simulated "shadow" book for comparing decisions.

## Project layout

| Module | Responsibility |
|---|---|
| `main.py` | Main trading loop: signals, filters, risk checks, execution, hedging |
| `config.py` | All parameters, overridable through environment variables |
| `mt5_connection.py` | MT5 initialisation and login (credentials from `.env`) |
| `data_handler.py`, `indicators.py`, `pair_selector.py` | Data access, indicators, choosing which pairs to trade |
| `ml_model.py`, `scheduler.py` | Fakeout classifier and its scheduled retraining |
| `rl_environment.py`, `train_rl.py` | Gymnasium environment and PPO training |
| `risk_manager.py`, `order_execution.py` | Risk limits, position sizing and order handling |
| `journal.py`, `shadow_sim.py`, `mirror_engine.py` | Logging and simulation helpers |

## Setup

Requires Windows with the MetaTrader 5 terminal installed (the `MetaTrader5` Python package only works on Windows) and an MT5 **demo** account.

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env      # then fill in your demo account details
python train_rl.py          # optional: train the PPO agent -> models/rl_agent.zip
python main.py
```

Credentials are read only from `.env`, which is git-ignored. Every strategy and risk parameter in `config.py` can be overridden through environment variables.

## Roadmap

These are ideas I'm planning, not features the bot has today:

- A backtesting harness and walk-forward parameter optimisation
- Kelly-criterion position sizing (its parameters exist in `config.py` but are not yet used)
- A global kill switch that closes all positions
- Unit tests for the indicator and risk modules
