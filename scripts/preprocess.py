import pandas as pd

import joblib
import os
from sklearn.preprocessing import LabelEncoder

def process_data(input_path, output_path, model_dir='models'):
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    df = pd.read_csv(input_path)
    
    df['finish_position'] = pd.to_numeric(df['finish_position'], errors='coerce').fillna(20)
    df['grid_position'] = pd.to_numeric(df['grid_position'], errors='coerce').fillna(20)
    
    # Driver Form (Avg finish last 3 races)
    df = df.sort_values(['year', 'round'])
    df['driver_form'] = df.groupby('driver')['finish_position'] \
                          .transform(lambda x: x.shift(1).rolling(3, min_periods=1).mean()).fillna(15)
    
    le_driver = LabelEncoder()
    le_team = LabelEncoder()
    le_circuit = LabelEncoder()
    
    df['driver_id'] = le_driver.fit_transform(df['driver'])
    df['team_id'] = le_team.fit_transform(df['team'])
    df['circuit_id'] = le_circuit.fit_transform(df['circuit_name'])
    
    joblib.dump(le_driver, f'{model_dir}/le_driver.pkl')
    joblib.dump(le_team, f'{model_dir}/le_team.pkl')
    joblib.dump(le_circuit, f'{model_dir}/le_circuit.pkl')
    
    # Filter columns
    final_df = df[[
        'year', 'round', 'circuit_id', 'driver_id', 'team_id',
        'fp_pos', 'driver_form', 'air_temp', 'rainfall',
        'grid_position', 'finish_position', 'safety_car'
    ]]
    
    final_df.to_csv(output_path, index=False)
    print("Preprocessing complete. Encoders saved to /models")

if __name__ == "__main__":
    process_data('data/raw/f1_raw_data.csv', 'data/processed/f1_training_data.csv')