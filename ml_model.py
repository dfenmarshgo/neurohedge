# ml_model.py
import os
import joblib
import numpy as np
import logging
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from config import ML_MODEL_PATH

class FakeoutClassifier:
    def __init__(self):
        # Create a new GradientBoostingClassifier instance.
        self.model = GradientBoostingClassifier()
        self.fitted = False  # Flag to track whether the model has been trained.
    
    def train(self, X, y):
        """
        Train the GradientBoostingClassifier on features X and labels y.
        """
        self.model.fit(X, y)
        self.fitted = True
        logging.info("Training complete. Model is now fitted.")
    
    def predict(self, X):
        """
        Predict the probability for class 1 (fakeout) for each sample in X.
        If the model is not fitted, return a default probability of 0.5 for all samples.
        """
        if not self.fitted:
            logging.warning("Model is not fitted; returning default prediction of 0.5.")
            return np.full(len(X), 0.5)
        return self.model.predict_proba(X)[:, 1]
    
    def save_model(self, path=ML_MODEL_PATH):
        """
        Save the model and its fitted status to the specified path.
        """
        joblib.dump((self.model, self.fitted), path)
        logging.info("Model saved to %s.", path)
    
    def load_model(self, path=ML_MODEL_PATH):
        """
        Load the model and its fitted status from the specified path.
        If the file is not found, log that a new (unfitted) model will be used.
        """
        if os.path.exists(path):
            self.model, self.fitted = joblib.load(path)
            logging.info("Model loaded from %s.", path)
        else:
            logging.info("No model file found, using a new (unfitted) model.")
            self.fitted = False

def retrain_from_trade_history(csv_file):
    """
    Load trade data from a CSV file, prepare features and labels, and retrain the classifier.
    
    Feature extraction example:
      - Encode moving_average_direction as: 'up' -> 1, 'down' -> 0, others as 0.5.
      - Use volatility, ATR, encoded moving average direction, ML_confidence, and trade_size as features.
    
    Label:
      - Set to 1 if outcome is "win", otherwise 0.
    
    After training, the updated model is saved.
    """
    if not os.path.exists(csv_file):
        logging.error("Trade history file %s not found.", csv_file)
        return
    
    df = pd.read_csv(csv_file)
    
    # Encode moving_average_direction: up=1, down=0, neutral or others=0.5.
    df['MA_dir_encoded'] = df['moving_average_direction'].map({'up': 1, 'down': 0}).fillna(0.5)
    
    # Define feature columns; adjust these as needed.
    feature_cols = ['volatility', 'ATR', 'MA_dir_encoded', 'ML_confidence', 'trade_size']
    if not set(feature_cols).issubset(df.columns):
        logging.error("One or more required feature columns are missing in the trade history CSV.")
        return
    X = df[feature_cols].values
    
    # Define labels: win -> 1, loss -> 0.
    y = np.where(df['outcome'] == "win", 1, 0)
    
    # Create a new classifier, train it, and save the model.
    classifier = FakeoutClassifier()
    classifier.train(X, y)
    classifier.save_model()
    logging.info("Retraining from trade history complete.")

# Test harness (you can remove this in production)
if __name__ == "__main__":
    clf = FakeoutClassifier()
    # Create some dummy data to test predict
    X_dummy = np.random.randn(5, 10)
    print("Predictions:", clf.predict(X_dummy))
