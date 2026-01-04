import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from src.data_loader import FootballDataLoader
from src.features import FeatureEngineer
from src.config import settings
import matplotlib.pyplot as plt


def tune():
    print("Fetching data for tuning...")
    loader = FootballDataLoader()
    df = loader.load_data()

    if df.empty:
        print("Error: No data loaded.")
        return

    print(f"Preparing training set from {len(df)} matches...")
    engineer = FeatureEngineer()

    # Pre-calculate ELO
    df, _ = engineer.enrich_with_elo(df)

    X_list = []
    y_ou25 = []

    # Build dataset
    for index, row in df.iterrows():
        past_data = df[df['Date'] < row['Date']]
        if len(past_data) < 10: continue

        h_elo = row['HomeElo']
        a_elo = row['AwayElo']
        features = engineer.prepare_features(past_data, row['HomeTeam'], row['AwayTeam'], row['Date'], h_elo, a_elo)
        X_list.append(features)

        # Target: Over 2.5 Goals
        total_goals = row['FTHG'] + row['FTAG']
        y_ou25.append(1 if total_goals > 2.5 else 0)

    X = pd.DataFrame(X_list, columns=settings.FEATURES)
    y = pd.Series(y_ou25)

    print("\n--- 🧠 TUNING OVER/UNDER 2.5 MODEL ---")
    print("Testing different brain configurations...")

    # TimeSeriesSplit ensures we train on past and test on future (no cheating)
    tscv = TimeSeriesSplit(n_splits=5)

    # The settings we want to test
    param_dist = {
        'n_estimators': [50, 100, 150, 200],  # How many trees?
        'learning_rate': [0.01, 0.05, 0.1, 0.2],  # How fast to learn?
        'max_depth': [3, 4, 5, 6],  # How complex?
        'min_samples_split': [2, 5, 10],
        'subsample': [0.8, 0.9, 1.0]
    }

    model = GradientBoostingClassifier(random_state=42)

    # Randomly try 20 combinations
    search = RandomizedSearchCV(
        model,
        param_distributions=param_dist,
        n_iter=20,
        cv=tscv,
        scoring='accuracy',
        n_jobs=-1,
        random_state=42,
        verbose=1
    )

    search.fit(X, y)

    print(f"\n✅ Best Accuracy Found: {search.best_score_:.1%}")
    print("✅ Best Parameters:", search.best_params_)

    # --- FEATURE IMPORTANCE ---
    print("\n--- 📊 WHAT MATTERS MOST? (Feature Importance) ---")
    best_model = search.best_estimator_
    importances = best_model.feature_importances_
    indices = np.argsort(importances)[::-1]

    for f in range(len(settings.FEATURES)):
        print(f"{f + 1}. {settings.FEATURES[indices[f]]}: {importances[indices[f]]:.4f}")

    # Save Chart
    plt.figure(figsize=(12, 8))
    plt.title("Which Stats Drive the Predictions?")
    plt.bar(range(X.shape[1]), importances[indices], align="center")
    plt.xticks(range(X.shape[1]), [settings.FEATURES[i] for i in indices], rotation=90)
    plt.tight_layout()
    plt.savefig(settings.DATA_DIR / 'feature_importance.png')
    print(f"\nChart saved to: {settings.DATA_DIR / 'feature_importance.png'}")


if __name__ == "__main__":
    tune()
