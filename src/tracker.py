import json
import pandas as pd
from pathlib import Path
from datetime import datetime
from src.config import settings
from src.data_loader import FootballDataLoader

class PerformanceTracker:
    def __init__(self):
        self.log_file = settings.DATA_DIR / "prediction_log.json"
        self.metrics_file = settings.DATA_DIR / "performance_metrics.json"
        self.loader = FootballDataLoader()
        
        # Initialize log file if it doesn't exist
        if not self.log_file.exists():
            with open(self.log_file, 'w') as f:
                json.dump([], f)

    def log_prediction(self, match_data):
        """
        Logs a prediction for future verification.
        match_data: dict containing {date, home, away, market, prob, odds, stake, result=None}
        """
        try:
            with open(self.log_file, 'r') as f:
                history = json.load(f)
            
            # Check if already logged to avoid duplicates
            unique_id = f"{match_data['date']}_{match_data['home']}_{match_data['away']}_{match_data['market']}"
            if any(h.get('id') == unique_id for h in history):
                return

            match_data['id'] = unique_id
            match_data['status'] = 'pending'
            match_data['timestamp'] = datetime.now().isoformat()
            
            history.append(match_data)
            
            with open(self.log_file, 'w') as f:
                json.dump(history, f, indent=2)
                
        except Exception as e:
            print(f"Error logging prediction: {e}")

    def update_results(self):
        """
        Checks pending predictions against actual results in the data loader.
        """
        try:
            with open(self.log_file, 'r') as f:
                history = json.load(f)
            
            # Reload data to get latest results
            df = self.loader.load_data()
            if df.empty: return

            pending = [h for h in history if h['status'] == 'pending']
            if not pending: return

            updated_count = 0
            
            for pred in pending:
                # Find match in dataframe
                # We match on HomeTeam, AwayTeam and approximate Date
                match_date = pd.to_datetime(pred['matchDate']).date()
                
                match = df[
                    (df['HomeTeam'] == pred['home']) & 
                    (df['AwayTeam'] == pred['away']) & 
                    (df['Date'].dt.date == match_date)
                ]
                
                if not match.empty and pd.notna(match.iloc[0]['FTR']):
                    row = match.iloc[0]
                    result = self._check_bet_result(pred['market'], row)
                    
                    pred['status'] = 'settled'
                    pred['outcome'] = result # 'won' or 'lost'
                    pred['profit'] = self._calculate_profit(result, pred['odds'], pred['stake'])
                    updated_count += 1

            if updated_count > 0:
                with open(self.log_file, 'w') as f:
                    json.dump(history, f, indent=2)
                self._calculate_metrics(history)
                print(f"Updated {updated_count} predictions.")

        except Exception as e:
            print(f"Error updating results: {e}")

    def _check_bet_result(self, market, row):
        """Determines if a bet won or lost based on the match row."""
        ftr = row['FTR']
        goals = row['FTHG'] + row['FTAG']
        corners = row['HC'] + row['AC']
        cards = row['HY'] + row['AY'] + row['HR'] + row['AR']
        
        if market == "Home Win":
            return 'won' if ftr == 'H' else 'lost'
        elif market == "Away Win":
            return 'won' if ftr == 'A' else 'lost'
        elif market == "Over 1.5 Goals":
            return 'won' if goals > 1.5 else 'lost'
        elif market == "Over 2.5 Goals":
            return 'won' if goals > 2.5 else 'lost'
        elif market == "Over 9.5 Corners":
            return 'won' if corners > 9.5 else 'lost'
        elif market == "Over 3.5 Cards":
            return 'won' if cards > 3.5 else 'lost'
        
        return 'void'

    def _calculate_profit(self, result, odds, stake_percent):
        """Calculates profit/loss for a $1000 bankroll unit."""
        # Assume $1000 bankroll for tracking
        stake = 1000 * (stake_percent / 100)
        if result == 'won':
            return stake * (odds - 1)
        elif result == 'lost':
            return -stake
        return 0

    def _calculate_metrics(self, history):
        """Aggregates performance metrics."""
        settled = [h for h in history if h['status'] == 'settled']
        if not settled: return

        total_bets = len(settled)
        wins = len([h for h in settled if h['outcome'] == 'won'])
        total_profit = sum(h.get('profit', 0) for h in settled)
        roi = (total_profit / (total_bets * 100)) * 100 # Approx ROI based on flat unit, simplified

        metrics = {
            "total_bets": total_bets,
            "win_rate": round((wins / total_bets) * 100, 1),
            "total_profit": round(total_profit, 2),
            "roi": round(roi, 2),
            "last_updated": datetime.now().isoformat()
        }

        with open(self.metrics_file, 'w') as f:
            json.dump(metrics, f, indent=2)

    def get_metrics(self):
        if self.metrics_file.exists():
            with open(self.metrics_file, 'r') as f:
                return json.load(f)
        return {"total_bets": 0, "win_rate": 0, "total_profit": 0, "roi": 0}

# Test
if __name__ == "__main__":
    tracker = PerformanceTracker()
    tracker.update_results()
    print(tracker.get_metrics())