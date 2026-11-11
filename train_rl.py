# train_rl.py
import logging
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from rl_environment import NeuroHedgeEnv
import config

def train_rl_model(timeframe=config.TIMEFRAME, total_timesteps=50000, save_path="models/rl_agent.zip"):
    """
    Train the RL model using PPO.
    """
    logging.info("Starting RL model training")
    try:
        env = NeuroHedgeEnv(symbol=config.SYMBOL, timeframe=timeframe)
        check_env(env, warn=True)
        
        model = PPO(
            "MlpPolicy",
            env,
            verbose=1,
            tensorboard_log="./tensorboard/",
            learning_rate=0.0003,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2
        )
        
        model.learn(total_timesteps=total_timesteps, progress_bar=True)
        model.save(save_path)
        logging.info(f"RL model trained and saved to {save_path}")
        
        env.close()
    except Exception as e:
        logging.error(f"RL training failed: {e}")
        raise

if __name__ == "__main__":
    train_rl_model()