# rl_environment.py
import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces
from data_handler import scan_market
import logging

class NeuroHedgeEnv(gym.Env):
    """
    Custom Environment for Forex trading using stable-baselines3.
    """
    metadata = {'render.modes': ['human']}

    def __init__(self, symbol='EURUSD', timeframe=15, window_size=50, max_steps=1000):
        super(NeuroHedgeEnv, self).__init__()
        self.symbol = symbol
        self.timeframe = timeframe
        self.window_size = window_size
        self.max_steps = max_steps
        self.current_step = 0
        self.account_balance = 1000.0
        self.position = None
        self.entry_price = 0.0
        self.trade_size = 0.2
        self.max_drawdown = 0.05
        self.max_risk = 0.02

        self.action_space = spaces.Discrete(3)  # 0: hold, 1: buy, 2: sell
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(window_size, 6), dtype=np.float32
        )

        self.data = None
        self.current_data = None
        self.reset()

    def reset(self, seed=None):
        """
        Reset environment to initial state.
        """
        super().reset(seed=seed)
        self.current_step = 0
        self.account_balance = 1000.0
        self.position = None
        self.entry_price = 0.0
        self.trade_size = 0.2

        # Fetch market data
        market_data = scan_market(symbol=self.symbol, timeframe=self.timeframe, num_bars=self.max_steps + self.window_size)
        self.data = market_data['data']
        if self.data.empty:
            logging.error(f"No data for {self.symbol}, using dummy data")
            self.data = pd.DataFrame({
                'open': [1.0] * (self.max_steps + self.window_size),
                'high': [1.0] * (self.max_steps + self.window_size),
                'low': [1.0] * (self.max_steps + self.window_size),
                'close': [1.0] * (self.max_steps + self.window_size),
                'tick_volume': [100] * (self.max_steps + self.window_size)
            }, index=pd.date_range(start='2025-01-01', periods=self.max_steps + self.window_size, freq='T'))

        self.current_data = self.data.iloc[:self.window_size]
        return self._get_observation(), {}

    def _get_observation(self):
        """
        Get current state observation.
        """
        if len(self.current_data) < self.window_size:
            logging.warning("Insufficient data for observation, padding with zeros")
            pad_length = self.window_size - len(self.current_data)
            padded_data = pd.concat([
                pd.DataFrame(np.zeros((pad_length, len(self.current_data.columns))), columns=self.current_data.columns),
                self.current_data
            ])
        else:
            padded_data = self.current_data

        obs = np.array([
            padded_data['close'].values,
            padded_data['high'].values - padded_data['low'].values,
            padded_data['tick_volume'].values,
            (padded_data['close'] - padded_data['open']).values,
            padded_data['close'].pct_change().fillna(0).values,
            (padded_data['high'] - padded_data['low']) / padded_data['close'].shift(1).fillna(padded_data['close']).values
        ], dtype=np.float32).T

        if obs.shape[0] < self.window_size:
            obs = np.pad(obs, ((self.window_size - obs.shape[0], 0), (0, 0)), mode='constant')

        return obs[:self.window_size]

    def step(self, action):
        """
        Execute one time step within the environment.
        """
        self.current_step += 1
        done = self.current_step >= self.max_steps
        truncated = False
        reward = 0.0

        if done:
            return self._get_observation(), reward, done, truncated, {}

        current_price = self.data['close'].iloc[self.current_step + self.window_size - 1]
        self.current_data = self.data.iloc[self.current_step:self.current_step + self.window_size]

        # Execute action
        if action == 1 and self.position != 'long':  # Buy
            if self.position == 'short':
                reward += self._close_position(current_price)
            self.position = 'long'
            self.entry_price = current_price
            self.trade_size = min(self.trade_size, self.account_balance * self.max_risk / (current_price * 0.01))
        elif action == 2 and self.position != 'short':  # Sell
            if self.position == 'long':
                reward += self._close_position(current_price)
            self.position = 'short'
            self.entry_price = current_price
            self.trade_size = min(self.trade_size, self.account_balance * self.max_risk / (current_price * 0.01))
        elif action == 0 and self.position:  # Close position
            reward += self._close_position(current_price)
            self.position = None
            self.entry_price = 0.0

        # Update balance and check drawdown
        if self.position == 'long':
            unrealized_pnl = (current_price - self.entry_price) * self.trade_size * 100000
        elif self.position == 'short':
            unrealized_pnl = (self.entry_price - current_price) * self.trade_size * 100000
        else:
            unrealized_pnl = 0.0

        self.account_balance += unrealized_pnl
        drawdown = (1000.0 - self.account_balance) / 1000.0
        if drawdown > self.max_drawdown:
            reward -= 100.0
            done = True

        reward += unrealized_pnl * 0.1
        info = {'balance': self.account_balance, 'position': self.position}

        return self._get_observation(), reward, done, truncated, info

    def _close_position(self, current_price):
        """
        Close the current position and calculate reward.
        """
        if not self.position:
            return 0.0
        if self.position == 'long':
            profit = (current_price - self.entry_price) * self.trade_size * 100000
        else:
            profit = (self.entry_price - current_price) * self.trade_size * 100000
        self.account_balance += profit
        self.position = None
        self.entry_price = 0.0
        return profit * 0.1

    def render(self, mode='human'):
        """
        Render the environment state.
        """
        if mode == 'human':
            logging.info(f"Step: {self.current_step}, Balance: {self.account_balance:.2f}, "
                         f"Position: {self.position}, Entry: {self.entry_price:.5f}")