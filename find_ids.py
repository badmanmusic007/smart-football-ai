from src.data_loader import FootballDataLoader
from datetime import datetime

def get_upcoming_ids():
    loader = FootballDataLoader()
    
    # League 39 = Premier League
    # Season 2024 = 2024/2025 Season (Current Active Season)
    print("Fetching upcoming Premier League matches...")
    
    # This gets all matches for the season
    matches = loader.get_fixtures(league_id=39, season=2024)
    
    if not matches:
        print("Error: No matches found.")
        print("1. Check the [API ERROR] message above.")
        print("2. Ensure your API Key in .env is correct and saved.")
        return
    
    # Filter for matches in the future
    upcoming = []
    current_time = datetime.now().isoformat()
    
    for match in matches:
        if match['fixture']['date'] > current_time:
            upcoming.append(match)
    
    # Sort by date and show next 5
    upcoming.sort(key=lambda x: x['fixture']['date'])
    
    print("\n--- UPCOMING MATCH IDs ---")
    for match in upcoming[:5]:
        print(f"ID: {match['fixture']['id']} | {match['teams']['home']['name']} vs {match['teams']['away']['name']}")

if __name__ == "__main__":
    get_upcoming_ids()