from src.data_loader import FootballDataLoader
from datetime import datetime
import pandas as pd

def get_upcoming_ids():
    loader = FootballDataLoader()
    
    print("Fetching upcoming matches from local data...")
    
    # Load data
    df = loader.load_data()
    
    if df.empty:
        print("Error: No data found.")
        return
    
    # Filter for matches in the future (where FTHG is NaN and Date is >= today)
    today = pd.Timestamp.now().normalize()
    upcoming = df[(df['Date'] >= today) & (df['FTHG'].isna())].sort_values('Date')
    
    if upcoming.empty:
        print("No upcoming matches found in the current dataset.")
        return
    
    print("\n--- UPCOMING MATCHES ---")
    # Show next 10 matches
    for _, row in upcoming.head(10).iterrows():
        date_str = row['Date'].strftime('%Y-%m-%d')
        print(f"Date: {date_str} | {row['HomeTeam']} vs {row['AwayTeam']} ({row['Div']})")

if __name__ == "__main__":
    get_upcoming_ids()