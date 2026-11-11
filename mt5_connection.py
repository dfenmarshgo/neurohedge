# mt5_connection.py
import MetaTrader5 as mt5
import logging
import config
import os
from dotenv import load_dotenv

# Ensure .env is loaded
load_dotenv(override=True)

def initialize():
    """
    Initialize MT5 connection using credentials from config.py.
    """
    # Get credentials
    account = config.MT5_ACCOUNT_DEMO
    password = config.MT5_PASSWORD_DEMO
    server = config.MT5_SERVER

    # Debug environment
    env_account = os.getenv('MT5_ACCOUNT_DEMO')
    env_path = os.getenv('MT5_TERMINAL_PATH')
    dotenv_path = os.path.join(os.getcwd(), '.env')
    logging.info(f"Current working directory: {os.getcwd()}")
    logging.info(f".env file exists: {os.path.exists(dotenv_path)}")
    logging.info(f"Environment MT5_ACCOUNT_DEMO: {env_account}")
    logging.info(f"Config MT5_ACCOUNT_DEMO: {account}")
    logging.info(f"Environment MT5_TERMINAL_PATH: {env_path}")
    logging.info(f"Credentials: account={account}, password={'***' if password else None}, server={server}")

    # Validate account
    if str(account) == '0' or not env_account:
        logging.error("MT5_ACCOUNT_DEMO not set in .env or config")
        raise Exception("MT5 account not provided")

    try:
        account = int(account)
    except (ValueError, TypeError):
        logging.error(f"Invalid MT5 account number: {account}")
        raise Exception("MT5 account number must be an integer")

    logging.info(f"Attempting MT5 connection: Account {account}, Server {server}")

    # Clear cache
    mt5.shutdown()
    logging.info("MT5 shutdown called to clear cache")

    # Initialize
    if not mt5.initialize(login=account, password=password, server=server):
        logging.error(f"MT5 initialization failed: {mt5.last_error()}")
        raise Exception("MT5 initialization failed")

    # Verify login
    if not mt5.login(login=account, password=password, server=server):
        logging.error(f"MT5 login failed: {mt5.last_error()}")
        mt5.shutdown()
        raise Exception("MT5 login failed")

    account_info = mt5.account_info()
    if account_info is None:
        logging.error("Failed to retrieve account info")
        mt5.shutdown()
        raise Exception("Failed to retrieve account info")

    logging.info(f"MT5 connected: Account {account_info.login}, Balance {account_info.balance}")
    return True

def shutdown():
    """
    Shutdown MT5 connection.
    """
    mt5.shutdown()
    logging.info("MT5 connection closed")