import pandas as pd
import numpy as np
from src.model import MatchPredictor
from src.config import settings
from src.data_loader import FootballDataLoader
from src.features import FeatureEngineer
from datetime import datetime, timedelta
import os

def train_real_model():
    print("Fetching real historical data...")
    loader = FootballDataLoader()
    df = loader.load_data()
    
    if df.empty:
        print("Error: No data loaded.")
        return

    # Filter out future matches (where Result is NaN) for training
    df = df.dropna(subset=['FTR']).copy()

    print(f"Loaded {len(df)} matches. Preparing training set...")
    engineer = FeatureEngineer()
    
    # Pre-calculate ELO ratings for the entire history
    df, _ = engineer.enrich_with_elo(df)
    
    X_list = []
    y_res_list = []
    
    # Goal Lists
    y_ou15 = []
    y_ou25 = []
    y_ou35 = []
    y_ou45 = []
    y_btts = []
    y_corners = []
    y_cards = []
    
    # Define the cutoff for fetching historical weather (e.g., 90 days)
    historical_weather_cutoff = datetime.now() - timedelta(days=90)

    # Load existing model for feedback loop (Hard Mining)
    old_predictor = MatchPredictor()
    has_old_model = old_predictor.is_trained

    hard_examples_count = 0

    for index, row in df.iterrows():
        past_data = df[df['Date'] < row['Date']]
        if len(past_data) < 10: continue 
        
        # Pass the ELO ratings that were valid BEFORE this match was played
        h_elo = row['HomeElo']
        a_elo = row['AwayElo']
        div = row['Div']
        
        # Safely get referee, default to 'Unknown' if missing
        referee = row.get('Referee', 'Unknown')
        
        # Check if the match date is too old for the weather API
        fetch_weather = row['Date'].to_pydatetime() > historical_weather_cutoff
        
        features = engineer.prepare_features(
            past_data, row['HomeTeam'], row['AwayTeam'], row['Date'], div, referee, 
            h_elo, a_elo, fetch_weather=fetch_weather
        )
        
        # Match Result
        if row['FTR'] == 'H': y = 0
        elif row['FTR'] == 'D': y = 1
        else: y = 2
        
        # --- Feedback Loop: Hard Mining ---
        # If the old model was confidently wrong, duplicate this sample
        weight = 1
        if has_old_model:
            try:
                pred = old_predictor.predict(features)
                # Check if model was wrong on Match Winner
                probs = [pred['home_win_prob'], pred['draw_prob'], pred['away_win_prob']]
                predicted_class = np.argmax(probs)
                confidence = probs[predicted_class]
                
                if predicted_class != y and confidence > 0.70:
                    weight = 3 # Triple the weight for hard mistakes
                    hard_examples_count += 1
            except:
                pass # Ignore if prediction fails (e.g. feature mismatch)

        # Add to lists (weighted)
        for _ in range(weight):
            X_list.append(features)
            y_res_list.append(y)
            
            # Goal Targets
            total_goals = row['FTHG'] + row['FTAG']
            y_ou15.append(1 if total_goals > 1.5 else 0)
            y_ou25.append(1 if total_goals > 2.5 else 0)
            y_ou35.append(1 if total_goals > 3.5 else 0)
            y_ou45.append(1 if total_goals > 4.5 else 0)
            
            # BTTS Target
            y_btts.append(1 if (row['FTHG'] > 0 and row['FTAG'] > 0) else 0)
            
            # Corners Target
            total_corners = row['HC'] + row['AC']
            y_corners.append(1 if total_corners > 9.5 else 0)
            
            # Cards Target
            total_cards = row['HY'] + row['AY'] + row['HR'] + row['AR']
            y_cards.append(1 if total_cards > 3.5 else 0)
    
    if hard_examples_count > 0:
        print(f"🧠 Feedback Loop: Identified and boosted {hard_examples_count} 'hard' matches where the AI was previously wrong.")

    X_train = pd.DataFrame(X_list, columns=settings.FEATURES)
    
    print("Training models...")
    predictor = MatchPredictor()
    predictor.train(X_train, pd.Series(y_res_list), pd.Series(y_ou15), pd.Series(y_ou25), pd.Series(y_ou35), pd.Series(y_ou45), pd.Series(y_btts), pd.Series(y_corners), pd.Series(y_cards))
    print(f"Success! Models saved to: {settings.MODELS_DIR}")

if __name__ == "__main__":
    train_real_model()