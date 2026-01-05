from pathlib import Path

class Config:
    # Paths
    BASE_DIR = Path(__file__).resolve().parent.parent
    DATA_DIR = BASE_DIR / "data"
    MODELS_DIR = BASE_DIR / "models"
    
    # Create directories if they don't exist
    DATA_DIR.mkdir(exist_ok=True)
    MODELS_DIR.mkdir(exist_ok=True)

    # Model Settings
    FEATURES = [
        "home_form", "away_form", "home_home_form", "away_away_form", "h2h_home_wins", 
        "home_goals_scored", "away_goals_scored", "home_goals_conceded", "away_goals_conceded", 
        "home_elo", "away_elo", 
        "home_shots", "away_shots", "home_shots_ot", "away_shots_ot", 
        "home_rest_days", "away_rest_days", 
        "home_corners", "away_corners", 
        "home_cards", "away_cards",
        "referee_harshness",
        "home_shot_conversion", "away_shot_conversion",
        "home_save_ratio", "away_save_ratio",
        "home_sos", "away_sos",
        "is_E0", "is_E1", "is_SP1", "is_D1", "is_I1", "is_F1", "is_N1", "is_P1", "is_SC0"
    ]
    RANDOM_SEED = 42

settings = Config()
