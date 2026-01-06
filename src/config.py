from pathlib import Path

class Config:
    # Paths
    BASE_DIR = Path(__file__).resolve().parent.parent
    DATA_DIR = BASE_DIR / "data"
    MODELS_DIR = BASE_DIR / "models"
    
    # Create directories if they don't exist
    DATA_DIR.mkdir(exist_ok=True)
    MODELS_DIR.mkdir(exist_ok=True)

    # Data URLs
    DATA_URLS = [
        # Upcoming Fixtures (All Leagues)
        "https://www.football-data.co.uk/fixtures.csv",

        # 2025/2026 Season (Current)
        "https://www.football-data.co.uk/mmz4281/2526/E0.csv",  # Premier League
        "https://www.football-data.co.uk/mmz4281/2526/E1.csv",  # Championship
        "https://www.football-data.co.uk/mmz4281/2526/E2.csv",  # League 1
        "https://www.football-data.co.uk/mmz4281/2526/E3.csv",  # League 2
        "https://www.football-data.co.uk/mmz4281/2526/SP1.csv", # La Liga
        "https://www.football-data.co.uk/mmz4281/2526/SP2.csv", # Segunda Division
        "https://www.football-data.co.uk/mmz4281/2526/D1.csv",  # Bundesliga
        "https://www.football-data.co.uk/mmz4281/2526/D2.csv",  # Bundesliga 2
        "https://www.football-data.co.uk/mmz4281/2526/I1.csv",  # Serie A
        "https://www.football-data.co.uk/mmz4281/2526/I2.csv",  # Serie B
        "https://www.football-data.co.uk/mmz4281/2526/F1.csv",  # Ligue 1
        "https://www.football-data.co.uk/mmz4281/2526/F2.csv",  # Ligue 2
        "https://www.football-data.co.uk/mmz4281/2526/N1.csv",  # Eredivisie
        "https://www.football-data.co.uk/mmz4281/2526/B1.csv",  # Jupiler League
        "https://www.football-data.co.uk/mmz4281/2526/P1.csv",  # Liga Portugal
        "https://www.football-data.co.uk/mmz4281/2526/T1.csv",  # Super Lig
        "https://www.football-data.co.uk/mmz4281/2526/G1.csv",  # Super League (Greece)
        "https://www.football-data.co.uk/mmz4281/2526/SC0.csv", # Premiership
        "https://www.football-data.co.uk/mmz4281/2526/SC1.csv", # Championship (Scotland)

        # 2024/2025 Season (History)
        "https://www.football-data.co.uk/mmz4281/2425/E0.csv",
        "https://www.football-data.co.uk/mmz4281/2425/E1.csv",
        "https://www.football-data.co.uk/mmz4281/2425/E2.csv",
        "https://www.football-data.co.uk/mmz4281/2425/E3.csv",
        "https://www.football-data.co.uk/mmz4281/2425/SP1.csv",
        "https://www.football-data.co.uk/mmz4281/2425/SP2.csv",
        "https://www.football-data.co.uk/mmz4281/2425/D1.csv",
        "https://www.football-data.co.uk/mmz4281/2425/D2.csv",
        "https://www.football-data.co.uk/mmz4281/2425/I1.csv",
        "https://www.football-data.co.uk/mmz4281/2425/I2.csv",
        "https://www.football-data.co.uk/mmz4281/2425/F1.csv",
        "https://www.football-data.co.uk/mmz4281/2425/F2.csv",
        "https://www.football-data.co.uk/mmz4281/2425/N1.csv",
        "https://www.football-data.co.uk/mmz4281/2425/B1.csv",
        "https://www.football-data.co.uk/mmz4281/2425/P1.csv",
        "https://www.football-data.co.uk/mmz4281/2425/T1.csv",
        "https://www.football-data.co.uk/mmz4281/2425/G1.csv",
        "https://www.football-data.co.uk/mmz4281/2425/SC0.csv",
        "https://www.football-data.co.uk/mmz4281/2425/SC1.csv",

        # 2023/2024 Season (History)
        "https://www.football-data.co.uk/mmz4281/2324/E0.csv",
        "https://www.football-data.co.uk/mmz4281/2324/E1.csv",
        "https://www.football-data.co.uk/mmz4281/2324/SP1.csv",
        "https://www.football-data.co.uk/mmz4281/2324/D1.csv",
        "https://www.football-data.co.uk/mmz4281/2324/I1.csv",
    ]

    # Model Settings
    FEATURES = [
        "home_form", "away_form", "home_home_form", "away_away_form", "h2h_home_wins", 
        "home_goals_scored", "away_goals_scored", "home_goals_conceded", "away_goals_conceded", 
        "home_elo", "away_elo", 
        "home_shots", "away_shots", "home_shots_ot", "away_shots_ot", 
        "home_rest_days", "away_rest_days", 
        "home_corners", "away_corners", 
        "home_cards", "away_cards",
        "home_shot_conversion", "away_shot_conversion",
        "home_save_ratio", "away_save_ratio",
        "home_sos", "away_sos",
        "rain_mm", "temp_c",
        "is_E0", "is_E1", "is_SP1", "is_D1", "is_I1", "is_F1", "is_N1", "is_P1", "is_SC0",
        "is_E2", "is_E3", "is_SP2", "is_D2", "is_I2", "is_F2", "is_B1", "is_T1", "is_G1", "is_SC1",
        "is_CL", "is_EL", "is_ECL", "is_FAC", "is_LC", "is_DFB", "is_CDR", "is_CDF", "is_CI"
    ]
    RANDOM_SEED = 42

settings = Config()
