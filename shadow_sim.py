# shadow_sim.py
import logging

class ShadowSimulator:
    def __init__(self):
        self.simulated_trades = []
    
    def simulate_trade(self, entry_price, stop_loss, take_profit, direction):
        trade = {
            'entry_price': entry_price,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'direction': direction,
            'result': None
        }
        self.simulated_trades.append(trade)
        logging.info(f"Simulated trade added: {trade}")
        return trade
    
    def update_trade_result(self, trade, result):
        trade['result'] = result
        logging.info(f"Trade result updated: {trade}")
