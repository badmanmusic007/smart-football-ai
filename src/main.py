from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, FileResponse
from pydantic import BaseModel
from src.config import settings
from src.data_loader import FootballDataLoader
from src.features import FeatureEngineer
from src.model import MatchPredictor
import logging
import pandas as pd
from datetime import timedelta
import numpy as np
import threading
from typing import List, Optional
from pathlib import Path

# Initialize Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Smart Football AI Predictor",
    description="AI-powered match outcome prediction API",
    version="0.2.6"
)

# Initialize components
data_loader = FootballDataLoader()
feature_engineer = FeatureEngineer()
predictor = MatchPredictor()

# Global status tracker
training_status = {"state": "idle", "error": None}


def run_background_training():
    """Runs training in a separate thread to avoid blocking startup."""
    global training_status
    import train_model  # Import here to avoid circular dependency
    training_status["state"] = "training"
    try:
        logger.info("Starting background training...")
        train_model.train_real_model()
        predictor.load_model()
        training_status["state"] = "completed"
        logger.info("Background training complete. Model loaded.")
    except Exception as e:
        training_status["state"] = "failed"
        training_status["error"] = str(e)
        logger.error(f"Background training failed: {e}")


@app.on_event("startup")
async def startup_event():
    if not predictor.is_trained:
        logger.info("Model not found. Initiating background training...")
        thread = threading.Thread(target=run_background_training)
        thread.start()
    else:
        logger.info("Pre-trained model found. AI is ready immediately.")


# --- Rivalry Logic Database ---
RIVALRIES = {
    tuple(sorted(["Arsenal", "Tottenham"])): "North London Derby",
    tuple(sorted(["Liverpool", "Everton"])): "Merseyside Derby",
    tuple(sorted(["Man City", "Man United"])): "Manchester Derby",
    tuple(sorted(["Liverpool", "Man United"])): "North West Derby",
    tuple(sorted(["Real Madrid", "Barcelona"])): "El Clásico",
    tuple(sorted(["AC Milan", "Inter"])): "Derby della Madonnina",
    tuple(sorted(["Celtic", "Rangers"])): "Old Firm Derby",
    tuple(sorted(["Roma", "Lazio"])): "Derby della Capitale",
    tuple(sorted(["Benfica", "Sporting CP"])): "Derby de Lisboa",
    tuple(sorted(["Ajax", "Feyenoord"])): "De Klassieker",
    tuple(sorted(["Bayern Munich", "Dortmund"])): "Der Klassiker",
    tuple(sorted(["Arsenal", "Chelsea"])): "London Derby",
    tuple(sorted(["Chelsea", "Tottenham"])): "London Derby",
    tuple(sorted(["Newcastle", "Sunderland"])): "Tyne-Wear Derby",
    tuple(sorted(["Man United", "Leeds"])): "Roses Rivalry"
}

LEAGUE_NAMES = {
    'E0': 'Premier League', 'E1': 'Championship', 'E2': 'League 1', 'E3': 'League 2',
    'SP1': 'La Liga', 'SP2': 'Segunda Division',
    'D1': 'Bundesliga', 'D2': 'Bundesliga 2',
    'I1': 'Serie A', 'I2': 'Serie B',
    'F1': 'Ligue 1', 'F2': 'Ligue 2',
    'N1': 'Eredivisie', 'B1': 'Jupiler League', 'P1': 'Liga Portugal',
    'T1': 'Super Lig', 'G1': 'Super League (Greece)',
    'SC0': 'Premiership', 'SC1': 'Championship (Scotland)',
    'CL': 'Champions League', 'EL': 'Europa League', 'ECL': 'Europa Conference League',
    'FAC': 'FA Cup', 'LC': 'League Cup', 'DFB': 'DFB Pokal', 'CDR': 'Copa del Rey',
    'CDF': 'Coupe de France', 'CI': 'Coppa Italia'
}


def get_recent_matches(df, team, n=5):
    mask = ((df['HomeTeam'] == team) | (df['AwayTeam'] == team)) & df['FTR'].notna()
    recent = df[mask].tail(n).iloc[::-1]
    history = []
    for _, row in recent.iterrows():
        if row['FTR'] == 'D': res = 'D'
        elif (row['HomeTeam'] == team and row['FTR'] == 'H') or (row['AwayTeam'] == team and row['FTR'] == 'A'): res = 'W'
        else: res = 'L'
        opponent = row['AwayTeam'] if row['HomeTeam'] == team else row['HomeTeam']
        score = f"{int(row['FTHG'])}-{int(row['FTAG'])}"
        history.append({"date": row['Date'].strftime("%d/%m"), "opponent": opponent, "score": score, "result": res})
    return history

def get_h2h_history(df, team1, team2, n=5):
    mask = (((df['HomeTeam'] == team1) & (df['AwayTeam'] == team2)) | ((df['HomeTeam'] == team2) & (df['AwayTeam'] == team1))) & df['FTR'].notna()
    recent = df[mask].tail(n).iloc[::-1]
    history = []
    for _, row in recent.iterrows():
        if row['FTR'] == 'D': res = 'D'
        elif (row['HomeTeam'] == team1 and row['FTR'] == 'H') or (row['AwayTeam'] == team1 and row['FTR'] == 'A'): res = 'W'
        else: res = 'L'
        score = f"{int(row['FTHG'])}-{int(row['FTAG'])}"
        history.append({"date": row['Date'].strftime("%d/%m/%y"), "home": row['HomeTeam'], "away": row['AwayTeam'], "score": score, "result": res})
    return history

def get_standings(df, division):
    if df.empty: return []
    latest_date = df['Date'].max()
    season_start = pd.Timestamp(year=latest_date.year, month=7, day=1) if latest_date.month >= 7 else pd.Timestamp(year=latest_date.year - 1, month=7, day=1)
    season_df = df[(df['Div'] == division) & (df['Date'] > season_start) & (df['FTR'].notna())]
    teams = {}
    for _, row in season_df.iterrows():
        h, a, hg, ag, res = row['HomeTeam'], row['AwayTeam'], row['FTHG'], row['FTAG'], row['FTR']
        if h not in teams: teams[h] = {'P': 0, 'W': 0, 'D': 0, 'L': 0, 'GF': 0, 'GA': 0, 'Pts': 0}
        if a not in teams: teams[a] = {'P': 0, 'W': 0, 'D': 0, 'L': 0, 'GF': 0, 'GA': 0, 'Pts': 0}
        teams[h]['P'] += 1; teams[a]['P'] += 1
        teams[h]['GF'] += hg; teams[h]['GA'] += ag
        teams[a]['GF'] += ag; teams[a]['GA'] += hg
        if res == 'H':
            teams[h]['W'] += 1; teams[h]['Pts'] += 3; teams[a]['L'] += 1
        elif res == 'A':
            teams[a]['W'] += 1; teams[a]['Pts'] += 3; teams[h]['L'] += 1
        else:
            teams[h]['D'] += 1; teams[h]['Pts'] += 1; teams[a]['D'] += 1; teams[a]['Pts'] += 1
    standings = [{'Team': team, **stats, 'GD': stats['GF'] - stats['GA']} for team, stats in teams.items()]
    standings.sort(key=lambda x: (x['Pts'], x['GD'], x['GF']), reverse=True)
    return standings

def get_matches_against_similar_elo(df, team_name, target_elo, n=5):
    mask = ((df['HomeTeam'] == team_name) | (df['AwayTeam'] == team_name)) & df['FTR'].notna()
    team_matches = df[mask].copy()
    if team_matches.empty: return []
    team_matches['OpponentElo'] = np.where(team_matches['HomeTeam'] == team_name, team_matches['AwayElo'], team_matches['HomeElo'])
    team_matches['EloDiff'] = abs(team_matches['OpponentElo'] - target_elo)
    similar = team_matches.sort_values(['EloDiff', 'Date'], ascending=[True, False]).head(n)
    history = []
    for _, row in similar.iterrows():
        if row['FTR'] == 'D': res = 'D'
        elif (row['HomeTeam'] == team_name and row['FTR'] == 'H') or (row['AwayTeam'] == team_name and row['FTR'] == 'A'): res = 'W'
        else: res = 'L'
        score = f"{int(row['FTHG'])}-{int(row['FTAG'])}"
        opponent = row['AwayTeam'] if row['HomeTeam'] == team_name else row['HomeTeam']
        history.append({"date": row['Date'].strftime("%d/%m/%y"), "opponent": opponent, "elo": int(row['OpponentElo']), "score": score, "result": res})
    return history

def get_rivalry_info(home, away):
    pair = tuple(sorted([home, away]))
    return {"name": RIVALRIES[pair], "impact": "High Intensity: Historical form may be less reliable."} if pair in RIVALRIES else None

def get_streaks(df, team_name):
    """Analyze last 10 games for streaks."""
    mask = ((df['HomeTeam'] == team_name) | (df['AwayTeam'] == team_name)) & df['FTR'].notna()
    recent = df[mask].tail(10).iloc[::-1] # Last 10, most recent first
    
    streaks = []
    
    # Win/Loss Streak
    wins = 0
    losses = 0
    for _, row in recent.iterrows():
        is_home = row['HomeTeam'] == team_name
        res = row['FTR']
        if (is_home and res == 'H') or (not is_home and res == 'A'):
            wins += 1
            losses = 0
        elif (is_home and res == 'A') or (not is_home and res == 'H'):
            losses += 1
            wins = 0
        else:
            break # Streak broken
            
    if wins >= 3: streaks.append({"text": f"🔥 {wins} Wins in a Row", "color": "#10b981"})
    if losses >= 3: streaks.append({"text": f"❄️ {losses} Losses in a Row", "color": "#64748b"})
    
    # Scoring Streak
    scored_streak = 0
    for _, row in recent.iterrows():
        goals = row['FTHG'] if row['HomeTeam'] == team_name else row['FTAG']
        if goals > 0: scored_streak += 1
        else: break
        
    if scored_streak >= 5: streaks.append({"text": f"⚽ Scored in {scored_streak} Straight", "color": "#3b82f6"})
    
    return streaks

def generate_badges(features, is_home, streaks=None):
    badges = [] if streaks is None else streaks
    form, venue_form = (features[0], features[2]) if is_home else (features[1], features[3])
    goals_scored, goals_conceded = (features[5], features[7]) if is_home else (features[6], features[8])
    
    if form >= 10: badges.append({"text": "🔥 On Fire", "color": "#ef4444"})
    elif form <= 3: badges.append({"text": "❄️ Ice Cold", "color": "#94a3b8"})
    if is_home and venue_form >= 10: badges.append({"text": "🏰 Home Fortress", "color": "#10b981"})
    if not is_home and venue_form <= 3: badges.append({"text": "😨 Road Struggles", "color": "#f59e0b"})
    if goals_scored >= 2.0: badges.append({"text": "🔫 Attack Machine", "color": "#3b82f6"})
    if goals_conceded <= 0.8: badges.append({"text": "🚌 Iron Defense", "color": "#8b5cf6"})
    return badges

def get_oracle_prediction(home_xg, away_xg):
    h_goals = np.random.poisson(max(home_xg, 0.1), 1000)
    a_goals = np.random.poisson(max(away_xg, 0.1), 1000)
    scores = {}
    for h, a in zip(h_goals, a_goals):
        s = f"{h}-{a}"; scores[s] = scores.get(s, 0) + 1
    return max(scores, key=scores.get)

def get_strategic_insights(home, away, home_elo, away_elo, rivalry, features):
    insights = []
    
    # 1. Derby Match
    if rivalry:
        insights.append({
            "icon": "⚔️",
            "text": f"This is the {rivalry['name']}. Derbies are often tighter and more aggressive. Consider 'Over Cards' or 'Draw' markets."
        })
        
    # 2. ELO Mismatch
    elo_diff = home_elo - away_elo
    if elo_diff > 200:
        insights.append({
            "icon": "⚖️",
            "text": f"Huge mismatch detected (+{int(elo_diff)} ELO). The 'Home Win' odds may be too low. Look for value in 'Asian Handicap -1.5'."
        })
    elif elo_diff < -200:
        insights.append({
            "icon": "⚖️",
            "text": f"Huge mismatch detected ({int(elo_diff)} ELO). The 'Away Win' is highly likely. Consider 'Away Team to Score in Both Halves'."
        })
        
    # 3. Goal Drought
    avg_goals = (features[5] + features[6]) / 2 # Avg goals scored by both
    if avg_goals < 1.0:
        insights.append({
            "icon": "🌵",
            "text": "Both teams have low scoring averages. This could be a tactical gridlock. 'Under 2.5 Goals' is a strong candidate."
        })
        
    # 4. Fortress vs Traveler
    home_home_form = features[2]
    away_away_form = features[3]
    if home_home_form >= 10 and away_away_form <= 3:
        insights.append({
            "icon": "🏰",
            "text": f"{home} is a fortress at home, while {away} struggles on the road. This amplifies the home advantage."
        })

    return insights

def sanitize_payload(data):
    if isinstance(data, dict): return {k: sanitize_payload(v) for k, v in data.items()}
    if isinstance(data, (list, tuple)): return [sanitize_payload(v) for v in data]
    if isinstance(data, (float, np.floating)): return 0.0 if np.isnan(data) or np.isinf(data) else float(data)
    if isinstance(data, (int, np.integer)): return int(data)
    if pd.isna(data): return None
    return data

class MatchResultRequest(BaseModel):
    home: str; away: str; date: str
class BulkResultRequest(BaseModel):
    matches: List[MatchResultRequest]

@app.get("/", response_class=HTMLResponse)
async def root():
    template_path = Path(__file__).parent / "templates" / "index.html"
    with open(template_path, "r", encoding="utf-8") as f: return HTMLResponse(content=f.read())

@app.get("/report-image")
async def get_report_image(strategy: str):
    path = settings.DATA_DIR / f"report_{strategy}.png"
    if not path.exists(): raise HTTPException(404, "Report image not found.")
    return FileResponse(path)

@app.get("/teams")
async def get_available_teams():
    df = data_loader.load_data()
    if df.empty: raise HTTPException(503, "Failed to load data.")
    return {"teams": sorted(list(set(df['HomeTeam'].unique()) | set(df['AwayTeam'].unique())))}

@app.get("/elo-history")
async def get_elo_history(home: str, away: str):
    df = data_loader.load_data()
    if df.empty: raise HTTPException(503, "Failed to load data.")
    
    # Ensure ELO is calculated
    df, _ = feature_engineer.enrich_with_elo(df)
    
    def get_team_elo(team_name):
        mask = ((df['HomeTeam'] == team_name) | (df['AwayTeam'] == team_name)) & df['FTR'].notna()
        matches = df[mask].tail(20) # Last 20 games
        history = []
        for _, row in matches.iterrows():
            elo = row['HomeElo'] if row['HomeTeam'] == team_name else row['AwayElo']
            history.append({"date": row['Date'].strftime("%d/%m"), "elo": int(elo)})
        return history

    return {
        "home": get_team_elo(home),
        "away": get_team_elo(away)
    }

@app.get("/goal-form")
async def get_goal_form(home: str, away: str):
    df = data_loader.load_data()
    if df.empty: raise HTTPException(503, "Failed to load data.")
    
    def get_team_goals(team_name):
        mask = ((df['HomeTeam'] == team_name) | (df['AwayTeam'] == team_name)) & df['FTR'].notna()
        matches = df[mask].tail(5) # Last 5 games
        history = []
        for _, row in matches.iterrows():
            if row['HomeTeam'] == team_name:
                scored, conceded = row['FTHG'], row['FTAG']
                opponent = row['AwayTeam']
            else:
                scored, conceded = row['FTAG'], row['FTHG']
                opponent = row['HomeTeam']
            history.append({"opponent": opponent, "scored": int(scored), "conceded": int(conceded)})
        return history

    return {
        "home": get_team_goals(home),
        "away": get_team_goals(away)
    }

@app.get("/predict")
async def predict_api(home: str, away: str):
    if training_status["state"] == "failed": raise HTTPException(500, f"Model training failed: {training_status['error']}")
    if not predictor.is_trained: raise HTTPException(503, "Model is still training. Please refresh shortly.")
    df = data_loader.load_data()
    if df.empty: raise HTTPException(503, "Failed to load data.")
    all_teams = set(df['HomeTeam'].unique()) | set(df['AwayTeam'].unique())
    if home not in all_teams: raise HTTPException(404, f"Team '{home}' not found.")
    if away not in all_teams: raise HTTPException(404, f"Team '{away}' not found.")
    
    def get_domestic_league(team_name):
        team_matches = df[(df['HomeTeam'] == team_name) | (df['AwayTeam'] == team_name)]
        if team_matches.empty: return None
        domestic_matches = team_matches[~team_matches['Div'].isin(['CL', 'EL', 'ECL', 'FAC', 'LC', 'DFB', 'CDR', 'CDF', 'CI'])]
        if not domestic_matches.empty: return domestic_matches['Div'].mode()[0]
        return team_matches.iloc[-1]['Div']

    df, elo_dict = feature_engineer.enrich_with_elo(df)
    home_elo, away_elo = elo_dict.get(home, 1500), elo_dict.get(away, 1500)
    
    home_domestic_div = get_domestic_league(home) or 'E0'
    away_domestic_div = get_domestic_league(away) or 'E0'
    
    features = feature_engineer.prepare_features(df, home, away, pd.Timestamp.now(), home_domestic_div, 'Unknown', home_elo, away_elo)
    pred = predictor.predict(features)
    
    home_standings = get_standings(df, home_domestic_div)
    for t in home_standings:
        if t['Team'] == home: t['RestDays'] = features[15]
        
    away_standings = get_standings(df, away_domestic_div) if away_domestic_div != home_domestic_div else home_standings
    for t in away_standings:
        if t['Team'] == away: t['RestDays'] = features[16]

    def to_odds(p): return round(1 / p, 2) if p > 0.01 else 99.0
    
    p_home, p_draw, p_away = pred["home_win_prob"], pred["draw_prob"], pred["away_win_prob"]
    p_1x = p_home + p_draw
    p_x2 = p_draw + p_away
    p_12 = p_home + p_away
    p_dnb_home = p_home / (p_home + p_away) if (p_home + p_away) > 0 else 0
    p_dnb_away = p_away / (p_home + p_away) if (p_home + p_away) > 0 else 0

    probs = {
        "Home Win": p_home, "Away Win": p_away, 
        "Over 1.5 Goals": pred["over_15_prob"], "Over 2.5 Goals": pred["over_25_prob"], 
        "Under 2.5 Goals": 1.0 - pred["over_25_prob"], 
        "Over 9.5 Corners": pred["corners_over_prob"], "Over 3.5 Cards": pred["cards_over_prob"],
        "Double Chance 1X": p_1x, "Double Chance X2": p_x2,
        "Draw No Bet Home": p_dnb_home, "Draw No Bet Away": p_dnb_away
    }
    
    best_market, best_prob = max(probs.items(), key=lambda item: item[1])
    recommendation = {"market": best_market, "probability": best_prob, "fair_odds": to_odds(best_prob)} if best_prob > 0.60 else None
    
    # Referee Factor
    ref_badge = None
    # ref_harshness = features[21] # Removed as per request
    
    # Trend Spotter
    home_streaks = get_streaks(df, home)
    away_streaks = get_streaks(df, away)
    
    # Strategic Insights
    rivalry = get_rivalry_info(home, away)
    strategic_insights = get_strategic_insights(home, away, home_elo, away_elo, rivalry, features)

    return sanitize_payload({
        "home_win": round(pred["home_win_prob"] * 100, 1), "draw": round(pred["draw_prob"] * 100, 1), "away_win": round(pred["away_win_prob"] * 100, 1),
        "rivalry": rivalry,
        "strategic_insights": strategic_insights,
        "features_used": {
            "home_form_score": features[0], "away_form_score": features[1],
            "home_home_form": features[2], "away_away_form": features[3],
            "home_goals_scored": round(features[5], 1), "away_goals_scored": round(features[6], 1),
            "home_goals_conceded": round(features[7], 1), "away_goals_conceded": round(features[8], 1),
            "home_elo": features[9], "away_elo": features[10],
            "home_shots_ot": round(features[13], 1), "away_shots_ot": round(features[14], 1),
            "home_corners": round(features[17], 1), "away_corners": round(features[18], 1),
            "home_cards": round(features[19], 1), "away_cards": round(features[20], 1),
            "home_rest_days": features[15], "away_rest_days": features[16]
        },
        "prediction": pred, "oracle_score": get_oracle_prediction((features[5] + features[8]) / 2, (features[6] + features[7]) / 2),
        "home_badges": generate_badges(features, True, home_streaks), "away_badges": generate_badges(features, False, away_streaks),
        "home_deep_dive": feature_engineer.calculate_deep_dive(df, home), "away_deep_dive": feature_engineer.calculate_deep_dive(df, away),
        "recommendation": recommendation,
        "fair_odds": {
            **{k: to_odds(v) for k, v in pred.items()},
            "dc_1x": to_odds(p_1x), "dc_x2": to_odds(p_x2), "dc_12": to_odds(p_12),
            "dnb_home": to_odds(p_dnb_home), "dnb_away": to_odds(p_dnb_away)
        },
        "home_history": get_recent_matches(df, home), "away_history": get_recent_matches(df, away), "h2h": get_h2h_history(df, home, away),
        "home_similar": get_matches_against_similar_elo(df, home, away_elo), "away_similar": get_matches_against_similar_elo(df, away, home_elo),
        "home_standings": home_standings, "away_standings": away_standings,
        "home_league_name": LEAGUE_NAMES.get(home_domestic_div, "Unknown"), "away_league_name": LEAGUE_NAMES.get(away_domestic_div, "Unknown")
    })

@app.post("/bulk-match-results")
async def get_bulk_match_results(request: BulkResultRequest):
    df = data_loader.load_data()
    results = []
    for r in request.matches:
        match = df[(df['HomeTeam'] == r.home) & (df['AwayTeam'] == r.away) & (pd.to_datetime(df['Date']).dt.date == pd.to_datetime(r.date).date())]
        if not match.empty and pd.notna(match.iloc[0]['FTR']):
            row = match.iloc[0]
            results.append({"home": r.home, "away": r.away, "result": {"ftr": row['FTR'], "fthg": int(row['FTHG']), "ftag": int(row['FTAG'])}})
    return results

@app.get("/scan")
async def scan_market():
    if training_status["state"] == "failed": return {"message": f"System Error: {training_status['error']}"}
    if not predictor.is_trained: return {"message": "Model is still training. Please try again shortly."}
    df = data_loader.load_data()
    if df.empty: return {"message": "Data could not be loaded."}

    today, next_week = pd.Timestamp.now().normalize(), pd.Timestamp.now().normalize() + timedelta(days=7)
    upcoming = df[(df['Date'] >= today) & (df['Date'] <= next_week) & (df['FTHG'].isna())]
    picks = []
    df_hist, elo_dict = feature_engineer.enrich_with_elo(df)

    for _, row in upcoming.iterrows():
        home, away = row['HomeTeam'], row['AwayTeam']
        if home not in elo_dict or away not in elo_dict: continue
        features = feature_engineer.prepare_features(df_hist[df_hist['Date'] < row['Date']], home, away, row['Date'], row['Div'], row['Referee'], elo_dict[home], elo_dict[away])
        pred = predictor.predict(features)
        
        market_map = {
            "Home Win": ("home_win_prob", "B365H", "AvgH"),
            "Away Win": ("away_win_prob", "B365A", "AvgA"),
            "Over 1.5 Goals": ("over_15_prob", "B365>2.5", "Avg>2.5"), # Proxy
            "Over 2.5 Goals": ("over_25_prob", "B365>2.5", "Avg>2.5")
        }

        for market, (prob_key, odds_key, avg_key) in market_map.items():
            prob = pred[prob_key]
            odds = row.get(odds_key)
            avg_odds = row.get(avg_key)

            if odds and odds > 1:
                value = (prob * odds) - 1
                sentiment = 0
                if avg_odds and avg_odds > 0:
                    sentiment = (avg_odds - odds) / avg_odds # Positive if B365 < Avg (Smart Money)

                if value > 0.05:
                    kelly_fraction = ((prob * odds) - 1) / (odds - 1)
                    picks.append({
                        "date": row['Date'].strftime("%a %d %b"),
                        "matchDate": row['Date'].isoformat(),
                        "league": LEAGUE_NAMES.get(row['Div'], row['Div']),
                        "match": f"{home} vs {away}",
                        "home": home, "away": away,
                        "market": market,
                        "prob": round(prob * 100, 1),
                        "bookie_odds": odds,
                        "value": round(value * 100, 1),
                        "kelly_stake": round(kelly_fraction * 100, 1),
                        "sentiment": round(sentiment * 100, 1)
                    })
    
    picks.sort(key=lambda x: x['value'], reverse=True)

    def build_acca(market_name, min_prob, target_odds, used_teams):
        candidates = sorted([p for p in picks if p['market'] == market_name and p['prob'] >= min_prob], key=lambda x: x['prob'], reverse=True)
        legs = []
        total_odds = 1.0
        league_counts = {}
        
        for cand in candidates:
            if total_odds >= target_odds: break
            
            league = cand['league']
            if league_counts.get(league, 0) >= 2: continue
            
            if cand['home'] not in used_teams and cand['away'] not in used_teams:
                legs.append(cand)
                total_odds *= cand['bookie_odds'] # Use real bookie odds for acca calculation
                used_teams.add(cand['home'])
                used_teams.add(cand['away'])
                league_counts[league] = league_counts.get(league, 0) + 1
                
        return {"legs": legs, "total_odds": total_odds} if legs else None

    master_used_teams = set()
    acca_super = build_acca("Over 1.5 Goals", 99.9, 2.00, master_used_teams)
    acca_o25 = build_acca("Over 2.5 Goals", 90, 1.50, master_used_teams)
    acca_o15 = build_acca("Over 1.5 Goals", 90, 1.50, master_used_teams)

    return sanitize_payload({"picks": picks[:15], "acca_super": acca_super, "acca_o25": acca_o25, "acca_o15": acca_o15})
