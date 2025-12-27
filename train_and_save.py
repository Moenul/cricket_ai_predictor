import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

DATA_PATH = r"./c411ae00-8da0-444e-9cd8-98460ee49ab5.csv.csv"

print("Loading data from:", DATA_PATH)

try:
    df = pd.read_csv(DATA_PATH)
except FileNotFoundError:
    print(f"\n❌ ERROR: File not found at path: {DATA_PATH}")
    print("Please ensure the file exists or update the DATA_PATH variable.")
    raise SystemExit()
    
req = {'bat_team','bowl_team','venue','overs','runs_last_5','wickets_last_5','total'}
missing = req - set(df.columns)
if missing:
    raise SystemExit(f"\n❌ ERROR: Missing required columns in CSV: {missing}")

print("Preprocessing data and fitting encoders...")
data = df[['bat_team','bowl_team','venue','overs','runs_last_5','wickets_last_5','total']].copy()

le_bat = LabelEncoder()
le_bowl = LabelEncoder()
le_venue = LabelEncoder()

data['bat_team_enc'] = le_bat.fit_transform(data['bat_team'])
data['bowl_team_enc'] = le_bowl.fit_transform(data['bowl_team'])
data['venue_enc'] = le_venue.fit_transform(data['venue'])

features = ['bat_team_enc','bowl_team_enc','venue_enc','overs','runs_last_5','wickets_last_5']
X = data[features]
y = data['total']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Data split: Training on {len(X_train)} samples, testing on {len(X_test)} samples.")
print("Training RandomForestRegressor (200 estimators)...")

#MODEL TRAINING
model = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
rmse = (mean_squared_error(y_test, y_pred))**0.5
print(f"✅ Training Complete. Test MAE: {mae:.3f}, RMSE: {rmse:.3f}")

print("Saving model and encoders...")

try:
    with open("model.pkl", "wb") as f:
        pickle.dump(model, f)

    with open("encoders.pkl", "wb") as f:
        pickle.dump({
            "bat": le_bat,
            "bowl": le_bowl,
            "venue": le_venue
        }, f)
    
    print("✅ Successfully saved model.pkl and encoders.pkl")

except Exception as e:
    print(f"\n❌ ERROR saving files: {e}")