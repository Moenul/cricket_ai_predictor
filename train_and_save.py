import os
import pickle
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor

# 1. Setup Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, "model.pkl")
enc_path = os.path.join(BASE_DIR, "encoders.pkl")

def train():
    # --- YOUR TRAINING LOGIC HERE ---
    # Example placeholder objects:
    # model = RandomForestRegressor().fit(X, y)
    # le_bat = LabelEncoder().fit(teams)
    
    print("Training model...")
    
    # After your training is done:
    try:
        with open(model_path, "wb") as f:
            pickle.dump(model, f)

        with open(enc_path, "wb") as f:
            pickle.dump({
                "bat": le_bat,
                "bowl": le_bowl,
                "venue": le_venue
            }, f)

        print(f"✅ Successfully saved model.pkl and encoders.pkl at {BASE_DIR}")
    except Exception as e:
        print(f"❌ ERROR saving files: {e}")

if __name__ == "__main__":
    train()