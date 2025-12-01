import fastf1
import pandas as pd
import os
import warnings

# Suppress FastF1 future warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

if not os.path.exists('cache'):
    os.makedirs('cache')

fastf1.Cache.enable_cache('cache')

def fetch_season_data(year):
    print(f"\n--- Fetching Season {year} ---")
    schedule = fastf1.get_event_schedule(year)
    races = schedule[schedule['EventFormat'] == 'conventional']
    
    season_data = []
    
    for _, event in races.iterrows():
        round_num = event['RoundNumber']
        event_name = event['EventName']
        print(f"Processing Round {round_num}: {event_name}")
        
        try:
            # Load Race Session (For Labels: Finish Position, SC)
            race_session = fastf1.get_session(year, round_num, 'R')
            race_session.load(telemetry=False, weather=True, messages=True)
            
            # Load Practice Session (For Features: FP Rank)
            fp_session = fastf1.get_session(year, round_num, 'FP2')
            fp_session.load(telemetry=False, weather=False, messages=False)
            
            # Create a dictionary of Driver -> FP2 Position
            fp_results = fp_session.results
            fp_map = {row['Abbreviation']: row['Position'] for _, row in fp_results.iterrows()}

            # Weather & SC 
            weather = race_session.weather_data.mean(numeric_only=True)
            sc_deployed = 0
            if 'TrackStatus' in race_session.laps.columns:
                statuses = race_session.laps['TrackStatus'].unique()
                if any(s in ['4', '6', '7'] for s in statuses): # 4=SC, 6/7=VSC
                    sc_deployed = 1

            # Build Dataset
            results = race_session.results
            for driver_code in results.index:
                row = results.loc[driver_code]
                driver_abbrev = row['Abbreviation']
                
                # Get FP2 Rank (Default to 20 if driver didn't run)
                fp_rank = fp_map.get(driver_abbrev, 20)

                season_data.append({
                    'year': year,
                    'round': round_num,
                    'circuit_name': event_name,
                    'driver': driver_abbrev,
                    'team': row['TeamName'],
                    'fp_pos': fp_rank,              
                    'grid_position': row['GridPosition'], 
                    'finish_position': row['Position'],   
                    'status': row['Status'],
                    'air_temp': weather['AirTemp'],
                    'rainfall': 1 if weather['Rainfall'] else 0,
                    'safety_car': sc_deployed
                })
                
        except Exception as e:
            print(f"Skipping {event_name}: {e}")
            continue

    return pd.DataFrame(season_data)

if __name__ == "__main__":
    # Collecting 2022, 2023, and 2024 data
    years = [2022, 2023, 2024]
    all_data = pd.DataFrame()
    
    for y in years:
        try:
            df = fetch_season_data(y)
            all_data = pd.concat([all_data, df])
            print(f"Successfully fetched {len(df)} records for {y}")
        except Exception as e:
            print(f"Failed to fetch {y}: {e}")
            print(f"Continuing with available data...")
            continue
    
    if len(all_data) > 0:
        all_data.to_csv('data/raw/f1_raw_data.csv', index=False)
        print(f"\nData collection complete! Saved {len(all_data)} total records.")
        print(f"Years included: {sorted(all_data['year'].unique().tolist())}")
    else:
        print("No data was collected.")