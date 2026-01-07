import requests
import logging
import time
from datetime import datetime
import json
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LiveDataConnector:
    def __init__(self, api_key="1ff06869922fb0a92ce84291805ee59a"):
        self.api_key = api_key
        self.base_url = "https://v3.football.api-sports.io"
        self.headers = {
            'x-rapidapi-host': "v3.football.api-sports.io",
            'x-rapidapi-key': self.api_key
        }
        self.cache_dir = Path(__file__).resolve().parent.parent / "data" / "cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Cache duration in seconds (e.g., 60 seconds for live data)
        self.cache_ttl = 60 

    def _get_cached_response(self, endpoint):
        """Simple file-based cache to save API calls."""
        cache_file = self.cache_dir / f"{endpoint.replace('/', '_')}.json"
        if cache_file.exists():
            if time.time() - cache_file.stat().st_mtime < self.cache_ttl:
                try:
                    with open(cache_file, 'r') as f:
                        return json.load(f)
                except:
                    pass # If read fails, ignore cache
        return None

    def _save_cached_response(self, endpoint, data):
        cache_file = self.cache_dir / f"{endpoint.replace('/', '_')}.json"
        with open(cache_file, 'w') as f:
            json.dump(data, f)

    def get_live_fixtures(self):
        """Fetch all currently live matches."""
        endpoint = "fixtures"
        params = {"live": "all"}
        
        # Check cache first
        cache_key = "live_fixtures"
        cached = self._get_cached_response(cache_key)
        if cached:
            return cached

        try:
            logger.info("Fetching LIVE fixtures from API-Football...")
            response = requests.get(f"{self.base_url}/{endpoint}", headers=self.headers, params=params)
            
            if response.status_code == 200:
                data = response.json().get('response', [])
                self._save_cached_response(cache_key, data)
                return data
            else:
                logger.error(f"API Error: {response.status_code} - {response.text}")
                return []
        except Exception as e:
            logger.error(f"Connection Error: {e}")
            return []

    def get_todays_fixtures(self):
        """Fetch all fixtures for today."""
        endpoint = "fixtures"
        today = datetime.now().strftime("%Y-%m-%d")
        params = {"date": today}
        
        # Cache for 1 hour for daily schedule
        original_ttl = self.cache_ttl
        self.cache_ttl = 3600 
        
        cache_key = f"fixtures_{today}"
        cached = self._get_cached_response(cache_key)
        
        if cached:
            self.cache_ttl = original_ttl
            return cached

        try:
            logger.info(f"Fetching fixtures for {today}...")
            response = requests.get(f"{self.base_url}/{endpoint}", headers=self.headers, params=params)
            
            if response.status_code == 200:
                data = response.json().get('response', [])
                self._save_cached_response(cache_key, data)
                self.cache_ttl = original_ttl
                return data
            else:
                self.cache_ttl = original_ttl
                return []
        except Exception as e:
            self.cache_ttl = original_ttl
            logger.error(f"Connection Error: {e}")
            return []

    def get_live_odds(self, fixture_id):
        """Fetch live odds for a specific fixture."""
        # Note: Live odds endpoint might be restricted on some plans, 
        # falling back to pre-match if needed or handling gracefully.
        endpoint = "odds/live"
        params = {"fixture": fixture_id}
        
        try:
            response = requests.get(f"{self.base_url}/{endpoint}", headers=self.headers, params=params)
            if response.status_code == 200:
                return response.json().get('response', [])
            return []
        except:
            return []

# Test block
if __name__ == "__main__":
    connector = LiveDataConnector()
    live = connector.get_live_fixtures()
    print(f"Found {len(live)} live matches.")
    if live:
        print(f"Sample: {live[0]['teams']['home']['name']} vs {live[0]['teams']['away']['name']} ({live[0]['goals']['home']}-{live[0]['goals']['away']})")
