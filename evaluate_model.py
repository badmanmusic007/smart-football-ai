import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from src.data_loader import FootballDataLoader
from src.features import FeatureEngineer
from src.model import MatchPredictor
from src.config import settings
from sklearn.ensemble import GradientBoostingClassifier


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
    odds_list = []
    y_ou25_list = []
    odds_ou25_list = []
    y_btts_list = []
    odds_btts_list = []

    # Generate features for all matches
    for index, row in df.iterrows():
        # Use data strictly before this match date to simulate real prediction conditions
        past_data = df[df['Date'] < row['Date']]

        # Need at least 10 games of history to make a decent guess
        if len(past_data) < 10: continue

        # Pass the ELO ratings that were valid BEFORE this match was played
        h_elo = row['HomeElo']
        a_elo = row['AwayElo']
        features = engineer.prepare_features(past_data, row['HomeTeam'], row['AwayTeam'], row['Date'], h_elo, a_elo)
        X_list.append(features)

        # Target: 0=Home, 1=Draw, 2=Away
        if row['FTR'] == 'H':
            y = 0
        elif row['FTR'] == 'D':
            y = 1
        else:
            y = 2
        y_list.append(y)

        # Store odds for simulation (Home, Draw, Away)
        odds_list.append([row['B365H'], row['B365D'], row['B365A']])

        # Target O/U 2.5: 1 if Over, 0 if Under
        total_goals = row['FTHG'] + row['FTAG']
        y_ou25_list.append(1 if total_goals > 2.5 else 0)

        # Store odds for O/U (Under, Over) - Note: Index 0=Under, 1=Over to match model output
        odds_ou25_list.append([row['B365<2.5'], row['B365>2.5']])

        # Target BTTS: 1 if Yes, 0 if No
        btts_yes = 1 if (row['FTHG'] > 0 and row['FTAG'] > 0) else 0
        y_btts_list.append(btts_yes)

        # Store odds for BTTS (No, Yes) - Index 0=No, 1=Yes
        odds_btts_list.append([row['B365BTTSN'], row['B365BTTSY']])

    X = pd.DataFrame(X_list, columns=settings.FEATURES)
    y = pd.Series(y_list)
    odds_series = pd.Series(odds_list)
    y_ou = pd.Series(y_ou25_list)
    odds_ou_series = pd.Series(odds_ou25_list)
    y_btts = pd.Series(y_btts_list)
    odds_btts_series = pd.Series(odds_btts_list)

    # Split: Train on the first 80% of matches, Test on the recent 20%
    split_index = int(len(X) * 0.8)

    X_train = X.iloc[:split_index]
    y_train = y.iloc[:split_index]
    X_test = X.iloc[split_index:]
    y_test = y.iloc[split_index:]
    odds_test = odds_series.iloc[split_index:]

    # O/U Split
    y_ou_train = y_ou.iloc[:split_index]
    y_ou_test = y_ou.iloc[split_index:]
    odds_ou_test = odds_ou_series.iloc[split_index:]

    # BTTS Split
    y_btts_train = y_btts.iloc[:split_index]
    y_btts_test = y_btts.iloc[split_index:]
    odds_btts_test = odds_btts_series.iloc[split_index:]

    print(f"\nTraining on {len(X_train)} older matches...")
    print(f"Testing on {len(X_test)} recent matches...")

    # --- MODEL 1: MATCH WINNER ---
    print("\nEvaluating Match Winner Model...")
    model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
    model.fit(X_train, y_train)

    probs = model.predict_proba(X_test)
    predictions = np.argmax(probs, axis=1)
    accuracy = accuracy_score(y_test, predictions)
    print(f"Match Winner Accuracy: {accuracy:.1%}")

    # --- MODEL 2: OVER/UNDER 2.5 ---
    print("Evaluating Over/Under 2.5 Model...")
    model_ou = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
    model_ou.fit(X_train, y_ou_train)

    probs_ou = model_ou.predict_proba(X_test)
    predictions_ou = np.argmax(probs_ou, axis=1)
    accuracy_ou = accuracy_score(y_ou_test, predictions_ou)
    print(f"O/U 2.5 Accuracy:      {accuracy_ou:.1%}")

    # --- MODEL 3: BTTS ---
    print("Evaluating BTTS Model...")
    model_btts = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
    model_btts.fit(X_train, y_btts_train)

    probs_btts = model_btts.predict_proba(X_test)
    predictions_btts = np.argmax(probs_btts, axis=1)
    accuracy_btts = accuracy_score(y_btts_test, predictions_btts)
    print(f"BTTS Accuracy:         {accuracy_btts:.1%}")

    # Reset indexes for clean iteration
    y_test_reset = y_test.reset_index(drop=True)
    odds_test_reset = odds_test.reset_index(drop=True)

    # --- SIMULATION LOOP ---
    thresholds = [0.0, 0.4, 0.5, 0.6, 0.7]

    print("\n--- 💰 STRATEGY 1: MATCH WINNER ROI ---")

    for threshold in thresholds:
        balance = 0
        bets_placed = 0
        bets_won = 0
        balance_history = []

        for i, prob_set in enumerate(probs):
            # Find the AI's top choice
            prediction = np.argmax(prob_set)
            confidence = prob_set[prediction]

            # Only bet if confidence is higher than threshold
            if confidence >= threshold:
                actual = y_test_reset[i]
                match_odds = odds_test_reset[i]
                odds_value = match_odds[prediction]

                # Skip if no valid odds available (Data missing)
                if odds_value <= 1.0:
                    continue

                bets_placed += 1

                if prediction == actual:
                    profit = (10 * odds_value) - 10
                    balance += profit
                    bets_won += 1
                else:
                    balance -= 10

            if bets_placed > 0:
                balance_history.append(balance)

        if bets_placed > 0:
            roi = (balance / (bets_placed * 10)) * 100
            print(
                f"Threshold {int(threshold * 100)}%+: {bets_placed} Bets | Win Rate: {bets_won / bets_placed:.1%} | Profit: ${balance:.2f} | ROI: {roi:.2f}%")

            if threshold == 0.0:  # Plot for the full strategy
                plt.figure(figsize=(10, 5))
                plt.plot(balance_history, label='Strategy 1: Match Winner')
                plt.title('Match Winner Bankroll (All Bets)')
                plt.xlabel('Bets Placed')
                plt.ylabel('Profit ($)')
                plt.grid(True)
                plt.savefig(settings.DATA_DIR / 'report_match_winner.png')
                plt.close()
        else:
            print(f"Threshold {int(threshold * 100)}%+: No bets placed.")

    # --- SIMULATION LOOP O/U ---
    y_ou_test_reset = y_ou_test.reset_index(drop=True)
    odds_ou_test_reset = odds_ou_test.reset_index(drop=True)

    print("\n--- 💰 STRATEGY 2: OVER/UNDER 2.5 ROI ---")

    for threshold in thresholds:
        balance = 0
        bets_placed = 0
        bets_won = 0
        balance_history = []

        for i, prob_set in enumerate(probs_ou):
            prediction = np.argmax(prob_set)
            confidence = prob_set[prediction]

            if confidence >= threshold:
                actual = y_ou_test_reset[i]
                match_odds = odds_ou_test_reset[i]
                odds_value = match_odds[prediction]

                # Skip if no valid odds available
                if odds_value <= 1.0:
                    continue

                bets_placed += 1

                if prediction == actual:
                    profit = (10 * odds_value) - 10
                    balance += profit
                    bets_won += 1
                else:
                    balance -= 10

            if bets_placed > 0:
                balance_history.append(balance)

        if bets_placed > 0:
            roi = (balance / (bets_placed * 10)) * 100
            print(
                f"Threshold {int(threshold * 100)}%+: {bets_placed} Bets | Win Rate: {bets_won / bets_placed:.1%} | Profit: ${balance:.2f} | ROI: {roi:.2f}%")

            if threshold == 0.6:  # Plot for the profitable strategy (60%+)
                plt.figure(figsize=(10, 5))
                plt.plot(balance_history, color='green', label='Strategy 2: O/U 2.5')
                plt.title('Over/Under 2.5 Bankroll (60%+ Confidence)')
                plt.xlabel('Bets Placed')
                plt.ylabel('Profit ($)')
                plt.grid(True)
                plt.savefig(settings.DATA_DIR / 'report_over_under.png')
                plt.close()
        else:
            print(f"Threshold {int(threshold * 100)}%+: No bets placed.")

    # --- SIMULATION LOOP BTTS ---
    y_btts_test_reset = y_btts_test.reset_index(drop=True)
    odds_btts_test_reset = odds_btts_test.reset_index(drop=True)

    print("\n--- 💰 STRATEGY 3: BTTS ROI ---")

    for threshold in thresholds:
        balance = 0
        bets_placed = 0
        bets_won = 0
        balance_history = []

        for i, prob_set in enumerate(probs_btts):
            prediction = np.argmax(prob_set)
            confidence = prob_set[prediction]

            if confidence >= threshold:
                actual = y_btts_test_reset[i]
                match_odds = odds_btts_test_reset[i]
                odds_value = match_odds[prediction]

                # Skip if no valid odds available
                if odds_value <= 1.0:
                    continue

                bets_placed += 1

                if prediction == actual:
                    profit = (10 * odds_value) - 10
                    balance += profit
                    bets_won += 1
                else:
                    balance -= 10

            if bets_placed > 0:
                balance_history.append(balance)

        if bets_placed > 0:
            roi = (balance / (bets_placed * 10)) * 100
            print(
                f"Threshold {int(threshold * 100)}%+: {bets_placed} Bets | Win Rate: {bets_won / bets_placed:.1%} | Profit: ${balance:.2f} | ROI: {roi:.2f}%")

            if threshold == 0.0:  # Plot for full strategy
                plt.figure(figsize=(10, 5))
                plt.plot(balance_history, color='purple', label='Strategy 3: BTTS')
                plt.title('BTTS Bankroll (All Bets)')
                plt.xlabel('Bets Placed')
                plt.ylabel('Profit ($)')
                plt.grid(True)
                plt.savefig(settings.DATA_DIR / 'report_btts.png')
                plt.close()
        else:
            print(f"Threshold {int(threshold * 100)}%+: No bets placed.")


if __name__ == "__main__":
    evaluate()