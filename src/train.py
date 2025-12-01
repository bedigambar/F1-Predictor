import pandas as pd
import joblib
from xgboost import XGBRegressor, XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

def train_models():
    df = pd.read_csv('data/processed/f1_training_data.csv')
    
    # --- MODEL 1: QUALIFYING PREDICTOR ---
    # Features: Practice Pace (fp_pos), Driver Form, Team Strength, Track
    # Target: Grid Position
    X_quali = df[['fp_pos', 'driver_form', 'team_id', 'circuit_id', 'driver_id']]
    y_quali = df['grid_position']
    
    X_train_q, X_test_q, y_train_q, y_test_q = train_test_split(X_quali, y_quali, test_size=0.2, random_state=42)
    
    print("Training Quali Model...")
    quali_model = XGBRegressor(n_estimators=100, max_depth=4, learning_rate=0.1)
    quali_model.fit(X_train_q, y_train_q)
    print(f"Quali MAE: {mean_absolute_error(y_test_q, quali_model.predict(X_test_q)):.2f}")

    # --- MODEL 2: RACE PREDICTOR ---
    # Features: Grid Position (Predicted by Model 1), Driver Form, Track, Weather
    X_race = df[['grid_position', 'driver_form', 'team_id', 'circuit_id', 'driver_id', 'air_temp', 'rainfall']]
    y_race = df['finish_position']
    
    X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(X_race, y_race, test_size=0.2, random_state=42)
    
    print("Training Race Model...")
    race_model = XGBRegressor(n_estimators=100, max_depth=5, learning_rate=0.05)
    race_model.fit(X_train_r, y_train_r)
    print(f"Race MAE: {mean_absolute_error(y_test_r, race_model.predict(X_test_r)):.2f}")

    # --- MODEL 3: SAFETY CAR ---
    X_sc = df[['circuit_id', 'rainfall', 'air_temp']]
    y_sc = df['safety_car']
    
    print("Training Safety Car Model...")
    sc_model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    sc_model.fit(X_sc, y_sc)
    
    # Saving models
    joblib.dump(quali_model, 'models/quali_model.pkl')
    joblib.dump(race_model, 'models/race_model.pkl')
    joblib.dump(sc_model, 'models/sc_model.pkl')
    print("All models saved.")

if __name__ == "__main__":
    train_models()