import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Config:
    PROJECT_ROOT = Path(__file__).parent.parent
    DATA_DIR = PROJECT_ROOT / "data"
    MODELS_DIR = PROJECT_ROOT / "models"
    
    # Create directories if they don't exist
    DATA_DIR.mkdir(exist_ok=True)
    MODELS_DIR.mkdir(exist_ok=True)

    # Open Data Sources (Football-Data.co.uk)
    # E0 = English Premier League
    DATA_URLS = [
        # Upcoming Fixtures (All Leagues)
        "https://www.football-data.co.uk/fixtures.csv",

        # 2025/2026 Season (Current)
        "https://www.football-data.co.uk/mmz4281/2526/E0.csv",  # Premier League
        "https://www.football-data.co.uk/mmz4281/2526/SP1.csv", # La Liga
        "https://www.football-data.co.uk/mmz4281/2526/D1.csv",  # Bundesliga
        "https://www.football-data.co.uk/mmz4281/2526/I1.csv",  # Serie A
        "https://www.football-data.co.uk/mmz4281/2526/F1.csv",  # Ligue 1
        "https://www.football-data.co.uk/mmz4281/2526/N1.csv",  # Eredivisie
        "https://www.football-data.co.uk/mmz4281/2526/P1.csv",  # Liga Portugal
        "https://www.football-data.co.uk/mmz4281/2526/SC0.csv", # Scotland Premiership

        # 2024/2025 Season (History)
        "https://www.football-data.co.uk/mmz4281/2425/E0.csv",
        "https://www.football-data.co.uk/mmz4281/2425/SP1.csv",
        "https://www.football-data.co.uk/mmz4281/2425/D1.csv",
        "https://www.football-data.co.uk/mmz4281/2425/I1.csv",
        "https://www.football-data.co.uk/mmz4281/2425/F1.csv",
        "https://www.football-data.co.uk/mmz4281/2425/N1.csv",
        "https://www.football-data.co.uk/mmz4281/2425/P1.csv",
        "https://www.football-data.co.uk/mmz4281/2425/SC0.csv",

        # 2023/2024 Season (History)
        "https://www.football-data.co.uk/mmz4281/2324/E0.csv",
        "https://www.football-data.co.uk/mmz4281/2324/SP1.csv",
        "https://www.football-data.co.uk/mmz4281/2324/D1.csv",
        "https://www.football-data.co.uk/mmz4281/2324/I1.csv",
        "https://www.football-data.co.uk/mmz4281/2324/F1.csv",
        "https://www.football-data.co.uk/mmz4281/2324/N1.csv",
        "https://www.football-data.co.uk/mmz4281/2324/P1.csv",
        "https://www.football-data.co.uk/mmz4281/2324/SC0.csv",

        # 2022/2023 Season (History)
        "https://www.football-data.co.uk/mmz4281/2223/E0.csv",
        "https://www.football-data.co.uk/mmz4281/2223/SP1.csv",
        "https://www.football-data.co.uk/mmz4281/2223/D1.csv",
        "https://www.football-data.co.uk/mmz4281/2223/I1.csv",
        "https://www.football-data.co.uk/mmz4281/2223/F1.csv",
        "https://www.football-data.co.uk/mmz4281/2223/N1.csv",
        "https://www.football-data.co.uk/mmz4281/2223/P1.csv",
        "https://www.football-data.co.uk/mmz4281/2223/SC0.csv",
    ]

    # Model Settings
    # We are simplifying features to what is available in the CSVs (Form + Home Advantage)
    FEATURES = ["home_form", "away_form", "home_home_form", "away_away_form", "h2h_home_wins", "home_goals_scored", "away_goals_scored", "home_goals_conceded", "away_goals_conceded", "home_elo", "away_elo", "home_shots", "away_shots", "home_shots_ot", "away_shots_ot", "home_rest_days", "away_rest_days", "home_corners", "away_corners", "home_cards", "away_cards"]
    RANDOM_SEED = 42

settings = Config()