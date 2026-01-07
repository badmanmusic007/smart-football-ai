import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from src.data_loader import FootballDataLoader
from src.features import FeatureEngineer
from src.model import MatchPredictor
from src.config import settings
import xgboost as xgb
import json
from datetime import datetime

def evaluate():
    print("Loading data for evaluation...")
    loader = FootballDataLoader()
    df = loader.load_data()

    if df.empty:
        print("No data found.")
        return

    print(f"Processing {len(df)} matches... (This might take a minute)")
    engineer = FeatureEngineer()

    # Pre-calculate ELO ratings for the entire history
    df, _ = engineer.enrich_with_elo(df)

    X_list = []
    y_list = []
    
    # Generate features for all matches
    for index, row in df.iterrows():
        past_data = df[df['Date'] < row['Date']]
        if len(past_data) < 10: continue

        h_elo = row['HomeElo']
        a_elo = row['AwayElo']
        div = row['Div']
        referee = row.get('Referee', 'Unknown')
        
        # Use default weather for evaluation to be consistent with training
        features = engineer.prepare_features(
            past_data, row['HomeTeam'], row['AwayTeam'], row['Date'], div, referee, 
            h_elo, a_elo, fetch_weather=False
        )
        X_list.append(features)

        # Target: 0=Home, 1=Draw, 2=Away
        if row['FTR'] == 'H': y = 0
        elif row['FTR'] == 'D': y = 1
        else: y = 2
        y_list.append(y)

    X = pd.DataFrame(X_list, columns=settings.FEATURES)
    y = pd.Series(y_list)

    # Split: Train on the first 80% of matches, Test on the recent 20%
    split_index = int(len(X) * 0.8)

    X_train = X.iloc[:split_index]
    y_train = y.iloc[:split_index]
    X_test = X.iloc[split_index:]
    y_test = y.iloc[split_index:]

    print(f"\nTraining on {len(X_train)} older matches...")
    print(f"Testing on {len(X_test)} recent matches...")

    # --- MODEL 1: MATCH WINNER ---
    print("\nEvaluating Match Winner Model...")
    model = xgb.XGBClassifier(n_estimators=100, learning_rate=0.05, max_depth=4, eval_metric='mlogloss', random_state=42)
    model.fit(X_train, y_train)

    probs = model.predict_proba(X_test)
    predictions = np.argmax(probs, axis=1)
    accuracy = accuracy_score(y_test, predictions)
    print(f"Match Winner Accuracy: {accuracy:.1%}")
    
    # --- PERFORMANCE TRACKER ---
    log_file = settings.DATA_DIR / "performance_log.json"
    log_entry = {
        "date": datetime.now().isoformat(),
        "accuracy": float(accuracy),
        "test_samples": len(y_test)
    }
    
    history = []
    if log_file.exists():
        try:
            with open(log_file, 'r') as f:
                history = json.load(f)
        except:
            pass
            
    history.append(log_entry)
    
    # Keep only last 50 entries
    if len(history) > 50:
        history = history[-50:]
        
    with open(log_file, 'w') as f:
        json.dump(history, f, indent=4)
        
    print(f"✅ Performance logged to {log_file}")

if __name__ == "__main__":
    evaluate()