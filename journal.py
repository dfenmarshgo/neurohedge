# journal.py
import os
import logging
from datetime import datetime

class JournalWriter:
    def __init__(self):
        self.log_dir = os.path.abspath("trade_logs")
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
            logging.info(f"Created trade_logs directory at {self.log_dir}")

    def write_entry(self, entry):
        try:
            timestamp = datetime.now()
            date_str = timestamp.strftime("%Y%m%d_%H%M%S")
            log_file = os.path.join(self.log_dir, f"journal_{date_str}.md")
            with open(log_file, "a", encoding="utf-8") as f:
                f.write(f"{timestamp.strftime('%Y-%m-%d %H:%M:%S')} - {entry}\n")
            logging.info(f"Journal entry written to {log_file}")
        except Exception as e:
            logging.error(f"Error writing journal entry: {e}")