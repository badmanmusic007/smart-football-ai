from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, FileResponse
from src.config import settings
from src.data_loader import FootballDataLoader
from src.features import FeatureEngineer
from src.model import MatchPredictor
import logging
import pandas as pd
from datetime import timedelta
import numpy as np
import threading

# Initialize Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Smart Football AI Predictor",
    description="AI-powered match outcome prediction API",
    version="0.2.0"
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
    'E0': 'Premier League',
    'E1': 'Championship',
    'SP1': 'La Liga',
    'D1': 'Bundesliga',
    'I1': 'Serie A',
    'F1': 'Ligue 1',
    'N1': 'Eredivisie',
    'P1': 'Liga Portugal',
    'SC0': 'Premiership'
}


def get_recent_matches(df, team, n=5):
    """Get the last n matches for a specific team."""
    # Filter for played matches only (where FTR is not NaN)
    mask = ((df['HomeTeam'] == team) | (df['AwayTeam'] == team)) & df['FTR'].notna()
    recent = df[mask].tail(n).iloc[::-1]

    history = []
    for _, row in recent.iterrows():
        if row['FTR'] == 'D':
            res = 'D'
        elif (row['HomeTeam'] == team and row['FTR'] == 'H') or \
                (row['AwayTeam'] == team and row['FTR'] == 'A'):
            res = 'W'
        else:
            res = 'L'

        opponent = row['AwayTeam'] if row['HomeTeam'] == team else row['HomeTeam']
        score = f"{int(row['FTHG'])}-{int(row['FTAG'])}"

        history.append({
            "date": row['Date'].strftime("%d/%m"),
            "opponent": opponent,
            "score": score,
            "result": res
        })
    return history


def get_h2h_history(df, team1, team2, n=5):
    """Get the last n head-to-head matches."""
    # Filter for played matches only
    mask = (((df['HomeTeam'] == team1) & (df['AwayTeam'] == team2)) | \
           ((df['HomeTeam'] == team2) & (df['AwayTeam'] == team1))) & df['FTR'].notna()
    recent = df[mask].tail(n).iloc[::-1]

    history = []
    for _, row in recent.iterrows():
        # Determine result relative to team1
        if row['FTR'] == 'D':
            res = 'D'
        elif (row['HomeTeam'] == team1 and row['FTR'] == 'H') or \
                (row['AwayTeam'] == team1 and row['FTR'] == 'A'):
            res = 'W'
        else:
            res = 'L'

        score = f"{int(row['FTHG'])}-{int(row['FTAG'])}"
        history.append({
            "date": row['Date'].strftime("%d/%m/%y"),
            "home": row['HomeTeam'],
            "away": row['AwayTeam'],
            "score": score,
            "result": res
        })
    return history


def get_standings(df, division):
    """
    Calculate league table for the current season (2025/2026) for a specific division.
    """
    # Filter for current season (approximate start date July 2025)
    season_start = pd.Timestamp("2025-07-01")
    # FIX: Add a filter to only include played matches, preventing calculations on NaN scores
    season_df = df[(df['Div'] == division) & (df['Date'] > season_start) & (df['FTR'].notna())]

    teams = {}
    for _, row in season_df.iterrows():
        h, a = row['HomeTeam'], row['AwayTeam']
        hg, ag = row['FTHG'], row['FTAG']
        res = row['FTR']

        if h not in teams: teams[h] = {'P': 0, 'W': 0, 'D': 0, 'L': 0, 'GF': 0, 'GA': 0, 'Pts': 0}
        if a not in teams: teams[a] = {'P': 0, 'W': 0, 'D': 0, 'L': 0, 'GF': 0, 'GA': 0, 'Pts': 0}

        teams[h]['P'] += 1;
        teams[a]['P'] += 1
        teams[h]['GF'] += hg;
        teams[h]['GA'] += ag
        teams[a]['GF'] += ag;
        teams[a]['GA'] += hg

        if res == 'H':
            teams[h]['W'] += 1;
            teams[h]['Pts'] += 3
            teams[a]['L'] += 1
        elif res == 'A':
            teams[a]['W'] += 1;
            teams[a]['Pts'] += 3
            teams[h]['L'] += 1
        else:
            teams[h]['D'] += 1;
            teams[h]['Pts'] += 1
            teams[a]['D'] += 1;
            teams[a]['Pts'] += 1

    standings = []
    for team, stats in teams.items():
        stats['Team'] = team
        stats['GD'] = stats['GF'] - stats['GA']
        standings.append(stats)

    # Sort by Pts (desc), GD (desc), GF (desc)
    standings.sort(key=lambda x: (x['Pts'], x['GD'], x['GF']), reverse=True)
    return standings


def get_matches_against_similar_elo(df, team_name, target_elo, n=5):
    """
    Find historical matches where 'team_name' played against an opponent with ELO close to 'target_elo'.
    """
    # Filter for matches involving the team
    mask = ((df['HomeTeam'] == team_name) | (df['AwayTeam'] == team_name)) & df['FTR'].notna()
    team_matches = df[mask].copy()

    if team_matches.empty:
        return []

    # Calculate opponent ELO for each match
    team_matches['OpponentElo'] = np.where(
        team_matches['HomeTeam'] == team_name,
        team_matches['AwayElo'],
        team_matches['HomeElo']
    )

    # Calculate difference from target ELO
    team_matches['EloDiff'] = abs(team_matches['OpponentElo'] - target_elo)

    # Sort by EloDiff (closest strength) then Date (most recent)
    similar = team_matches.sort_values(['EloDiff', 'Date'], ascending=[True, False]).head(n)

    history = []
    for _, row in similar.iterrows():
        # Determine result for the specific team
        if row['FTR'] == 'D':
            res = 'D'
        elif (row['HomeTeam'] == team_name and row['FTR'] == 'H') or \
                (row['AwayTeam'] == team_name and row['FTR'] == 'A'):
            res = 'W'
        else:
            res = 'L'

        score = f"{int(row['FTHG'])}-{int(row['FTAG'])}"
        opponent = row['AwayTeam'] if row['HomeTeam'] == team_name else row['HomeTeam']

        history.append({
            "date": row['Date'].strftime("%d/%m/%y"),
            "opponent": opponent,
            "elo": int(row['OpponentElo']),
            "score": score,
            "result": res
        })
    return history


def get_rivalry_info(home, away):
    pair = tuple(sorted([home, away]))
    if pair in RIVALRIES:
        return {
            "name": RIVALRIES[pair],
            "impact": "High Intensity: Historical form may be less reliable in this derby context."
        }
    return None

def generate_badges(features, is_home):
    """
    Analyze features to generate personality badges for a team.
    Indices based on config.FEATURES list.
    """
    badges = []
    # Indices: 0=Form, 2=HomeForm, 3=AwayForm, 5/6=GoalsScored, 7/8=GoalsConceded, 
    # 15/16=Rest, 17/18=Corners
    form = features[0] if is_home else features[1]
    venue_form = features[2] if is_home else features[3]
    goals_scored = features[5] if is_home else features[6]
    goals_conceded = features[7] if is_home else features[8]
    corners = features[17] if is_home else features[18]
    rest = features[15] if is_home else features[16]
    
    if form >= 10: badges.append({"text": "🔥 On Fire", "color": "#ff4757"})
    elif form <= 3: badges.append({"text": "❄️ Ice Cold", "color": "#747d8c"})
    if is_home and venue_form >= 10: badges.append({"text": "🏰 Home Fortress", "color": "#2ed573"})
    if not is_home and venue_form <= 3: badges.append({"text": "😨 Road Struggles", "color": "#ffa502"})
    if goals_scored >= 2.0: badges.append({"text": "🔫 Attack Machine", "color": "#1e90ff"})
    if goals_conceded <= 0.8: badges.append({"text": "🚌 Iron Defense", "color": "#5352ed"})
    return badges

def sanitize_payload(data):
    """Recursively replace NaN/Infinity with 0.0 for JSON compliance."""
    if isinstance(data, dict):
        return {k: sanitize_payload(v) for k, v in data.items()}
    elif isinstance(data, (list, tuple)):
        return [sanitize_payload(v) for v in data]
    elif isinstance(data, (float, np.floating)):
        if np.isnan(data) or np.isinf(data):
            return 0.0
        return float(data)
    elif isinstance(data, (int, np.integer)):
        return int(data)
    elif pd.isna(data):
        return None
    return data


@app.get("/", response_class=HTMLResponse)
async def root():
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Smart Football AI</title>
        <style>
            :root { --primary: #1a73e8; --danger: #ea4335; --success: #34a853; }
            * { box-sizing: border-box; }
            body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; background-color: #f8f9fa; color: #333; }
            .container { background: white; padding: 30px; border-radius: 16px; box-shadow: 0 10px 30px rgba(0,0,0,0.08); text-align: center; }

            /* Input Styles */
            .selectors { display: flex; gap: 15px; justify-content: center; margin-bottom: 25px; align-items: center; }
            .team-input { padding: 12px; border-radius: 8px; border: 1px solid #ddd; width: 220px; font-size: 16px; }
            .swap-btn { background: none; border: none; font-size: 1.5em; cursor: pointer; color: #666; padding: 0 10px; transition: transform 0.2s; }
            .swap-btn:hover { color: #1a73e8; transform: scale(1.1); }
            button.main-btn { width: 100%; padding: 16px; background: var(--primary); color: white; border: none; border-radius: 8px; font-size: 18px; font-weight: bold; cursor: pointer; transition: 0.3s; }
            button.main-btn:hover { background: #1557b0; transform: translateY(-2px); }

            /* Animation */
            @keyframes slideUp {
                from { opacity: 0; transform: translateY(20px); }
                to { opacity: 1; transform: translateY(0); }
            }
            .animate-in { animation: slideUp 0.4s ease-out forwards; }

            /* Icon Jump Animation */
            @keyframes iconJump {
                0% { transform: scale(1); }
                50% { transform: scale(1.5); }
                100% { transform: scale(1); }
            }

            /* Rivalry Badge */
            .rivalry-card {
                background: linear-gradient(135deg, #ff416c, #ff4b2b);
                color: white; padding: 15px; border-radius: 10px; margin-bottom: 20px; font-weight: bold;
            }

            /* Collapsible Logic */
            .collapsible {
                background-color: #fff;
                color: #444;
                cursor: pointer;
                padding: 18px;
                width: 100%;
                border: none;
                text-align: left;
                outline: none;
                font-size: 1.1em;
                font-weight: bold;
                border-bottom: 1px solid #eee;
                border-radius: 8px;
                transition: 0.3s;
                display: flex;
                justify-content: space-between;
                align-items: center;
                margin-bottom: 10px;
                box-shadow: 0 2px 5px rgba(0,0,0,0.05);
            }
            .active, .collapsible:hover { background-color: #f8f9fa; }
            .collapsible:after { content: '\\002B'; color: #777; font-weight: bold; margin-left: 10px; font-size: 1.5em; display: block; }
            .collapsible:hover:after { animation: iconJump 0.3s ease-in-out; color: var(--primary); }
            .active:after { content: "\\2212"; }
            .content {
                padding: 0 18px;
                background-color: white;
                max-height: 0;
                overflow: hidden;
                transition: max-height 0.3s ease-out;
                border-radius: 0 0 8px 8px;
                margin-bottom: 15px;
            }
            .content.open { padding: 15px 18px; border: 1px solid #eee; border-top: none; box-shadow: 0 4px 6px rgba(0,0,0,0.05); }

            /* Match Rows */
            .match-row { display: flex; justify-content: space-between; padding: 8px 0; border-bottom: 1px solid #eee; font-size: 0.9em; }
            .res-badge { padding: 2px 6px; border-radius: 4px; font-weight: bold; font-size: 0.8em; color: white; min-width: 20px; text-align: center; display: inline-block; margin-right: 8px; }
            .res-W { background-color: var(--success); }
            .res-D { background-color: #9aa0a6; }
            .res-L { background-color: var(--danger); }

            /* Probabilities */
            .prob-grid { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 10px; margin: 20px 0; }
            .prob-box { background: #f8f9fa; padding: 15px; border-radius: 10px; border-top: 4px solid var(--primary); }

            /* Comparison Styles */
            .comp-row { display: flex; justify-content: space-between; align-items: center; padding: 8px 0; border-bottom: 1px solid #eee; }
            .comp-val { font-weight: bold; width: 30%; font-size: 1.1em; }
            .comp-val.home { text-align: right; color: #333; }
            .comp-val.away { text-align: left; color: #333; }
            .comp-lbl { font-size: 0.8em; color: #888; text-align: center; width: 40%; text-transform: uppercase; letter-spacing: 0.5px; }

            /* Team Badges */
            .team-badge { display: inline-block; font-size: 0.75em; padding: 2px 6px; border-radius: 4px; color: white; margin-right: 4px; margin-top: 4px; font-weight: bold; }

            /* History Grid */
            .history-container { display: grid; grid-template-columns: 1fr 1fr; gap: 15px; }
            @media (max-width: 600px) { 
                .history-container { grid-template-columns: 1fr; }
                .selectors { flex-direction: column; gap: 10px; }
                .team-input { width: 100%; }
                .swap-btn { transform: rotate(90deg); margin: 5px 0; }
                .container { padding: 20px; }
                body { padding: 10px; }
            }

            /* Scanner Styles */
            .scan-btn { background: #6f42c1; color: white; border: none; padding: 12px 20px; border-radius: 8px; font-weight: bold; cursor: pointer; width: 100%; margin-top: 10px; transition: 0.3s; }
            .scan-btn:hover { background: #5a32a3; }

            #result { display: none; margin-top: 30px; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>⚽ Smart Football AI</h1>
            <div class="selectors">
                <input type="text" id="homeTeam" class="team-input" list="teamList" placeholder="Home Team">
                <button class="swap-btn" onclick="swapTeams()" title="Swap Teams">⇄</button>
                <input type="text" id="awayTeam" class="team-input" list="teamList" placeholder="Away Team">
                <datalist id="teamList"><!-- Populated by JS --></datalist>
            </div>
            <button class="main-btn" onclick="predict()">Generate Analysis</button>
            <button class="scan-btn" onclick="scanMarket()">🔍 Scan for Value Bets (Next 7 Days)</button>

            <div id="result"></div>
            
            <div style="margin-top: 40px; border-top: 1px solid #eee; padding-top: 20px; text-align: center;">
                <h3 style="color:#555;">Backtesting Reports</h3>
                <button onclick="showReport('match_winner')" style="background:#6c757d; margin-right:10px; width:auto; padding: 10px 20px; color:white; border:none; border-radius:5px; cursor:pointer;">Match Winner ROI</button>
                <button onclick="showReport('over_under')" style="background:#6c757d; margin-right:10px; width:auto; padding: 10px 20px; color:white; border:none; border-radius:5px; cursor:pointer;">Over/Under 2.5 ROI</button>
                <button onclick="showReport('btts')" style="background:#6c757d; width:auto; padding: 10px 20px; color:white; border:none; border-radius:5px; cursor:pointer;">BTTS ROI</button>
            </div>
        </div>

        <script>
            // Load teams on startup
            fetch('/teams')
                .then(response => response.json())
                .then(data => {
                    const teams = data.teams;
                    const dataList = document.getElementById('teamList');
                    let options = '';
                    teams.forEach(team => {
                        options += `<option value="${team}">${team}</option>`;
                    });
                    dataList.innerHTML = options;
                    window.allTeams = teams;
                });

            function swapTeams() {
                const home = document.getElementById('homeTeam');
                const away = document.getElementById('awayTeam');
                const temp = home.value;
                home.value = away.value;
                away.value = temp;
            }

            function showReport(strategy) {
                const url = `/report-image?strategy=${strategy}`;
                window.open(url, '_blank');
            }

            function toggleCollapsible(element) {
                element.classList.toggle("active");
                var content = element.nextElementSibling;
                if (content.style.maxHeight){
                    content.style.maxHeight = null;
                    content.classList.remove("open");
                } else {
                    content.classList.add("open");
                    content.style.maxHeight = content.scrollHeight + "px";
                }
            }
            
            function renderTeamStanding(standings, teamName, leagueName) {
                const team = standings.find(t => t.Team === teamName);
                if (!team) return '<div class="stat-box" style="margin-bottom:15px; color:#999;">No League Data</div>';
                
                const rank = standings.indexOf(team) + 1;
                let rankColor = "#666";
                if (rank <= 4) rankColor = "#1a73e8"; // Top 4
                if (rank >= standings.length - 3) rankColor = "#ea4335"; // Relegation
                
                // Fatigue Badge
                let fatigueHTML = '';
                if (team.RestDays <= 3) {
                    fatigueHTML = '<span style="background:#fff3cd; color:#856404; font-size:0.7em; padding:2px 6px; border-radius:4px; margin-left:5px;">⚠️ Tired</span>';
                }
                
                return `
                    <div class="stat-box" style="margin-bottom: 15px; text-align: left; border-left: 4px solid ${rankColor}; background: #fff; padding:10px; border-radius:4px; border:1px solid #eee;">
                        <div style="font-size: 0.75em; color: #888; text-transform: uppercase; letter-spacing: 0.5px;">${leagueName} Position</div>
                        <div style="display: flex; justify-content: space-between; align-items: center; margin-top: 5px;">
                            <div style="font-size: 1.8em; font-weight: bold; color: ${rankColor};">#${rank}</div>
                            <div style="text-align: right;">
                                <div style="font-weight: bold; font-size: 1.1em;">${team.Pts} <span style="font-size:0.7em; color:#666;">PTS</span></div>
                                <div style="font-size: 0.8em; color: #666;">${team.P} Played</div>
                            </div>
                        </div>
                        <div style="font-size: 0.85em; margin-top: 8px; padding-top: 8px; border-top: 1px solid #eee; color: #555; display: flex; justify-content: space-between;">
                            <span><b>W:</b> ${team.W}</span> <span><b>D:</b> ${team.D}</span> <span><b>L:</b> ${team.L}</span> <span><b>Rest:</b> ${team.RestDays}d ${fatigueHTML}</span>
                        </div>
                    </div>
                `;
            }

            async function predict() {
                const home = document.getElementById('homeTeam').value;
                const away = document.getElementById('awayTeam').value;
                const resultDiv = document.getElementById('result');

                if(!home || !away) return alert("Select both teams");
                
                if (window.allTeams && !window.allTeams.includes(home)) {
                    alert(`Team '${home}' not found. Please select from the list.`);
                    return;
                }
                if (window.allTeams && !window.allTeams.includes(away)) {
                    alert(`Team '${away}' not found. Please select from the list.`);
                    return;
                }

                resultDiv.style.display = 'block';
                resultDiv.innerHTML = "Processing match data...";

                try {
                    const response = await fetch(`/predict?home=${encodeURIComponent(home)}&away=${encodeURIComponent(away)}`);
                    const data = await response.json();
                    
                    if (response.status !== 200) {
                        resultDiv.innerHTML = `<div style="color:red">Error: ${data.detail}</div>`;
                        return;
                    }

                    let rivalryHTML = data.rivalry ? `
                        <div class="rivalry-card animate-in">
                            🔥 ${data.rivalry.name} <br>
                            <span style="font-size:0.85em; font-weight:normal;">${data.rivalry.impact}</span>
                        </div>` : '';
                        
                    // Best Bet Logic
                    let bestBetHTML = '';
                    if (data.recommendation) {
                        const rec = data.recommendation;
                        bestBetHTML = `
                            <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 15px; border-radius: 12px; margin-bottom: 20px; box-shadow: 0 4px 15px rgba(0,0,0,0.2); text-align: center;">
                                <div style="font-size: 0.9em; text-transform: uppercase; letter-spacing: 1px; opacity: 0.9;">⭐ AI Top Pick</div>
                                <div style="font-size: 1.8em; font-weight: bold; margin: 5px 0;">${rec.market}</div>
                                <div style="font-size: 1.1em;">
                                    Probability: <b>${(rec.probability * 100).toFixed(1)}%</b>
                                    <span style="margin: 0 10px; opacity: 0.5;">|</span>
                                    Fair Odds: <b>${rec.fair_odds}</b>
                                </div>
                            </div>
                        `;
                    }
                    
                    // Goal Markets Table
                    const odds = data.fair_odds;
                    const pred = data.prediction;
                    const goalRow = (label, prob, odds) => {
                        const isLikely = prob > 55;
                        const style = isLikely ? 'background:#e8f0fe; color:#1967d2; font-weight:bold;' : '';
                        return `
                            <tr style="${style}">
                                <td style="padding:8px; border-bottom:1px solid #eee;">${label}</td>
                                <td style="padding:8px; border-bottom:1px solid #eee;">${prob}%</td>
                                <td style="padding:8px; border-bottom:1px solid #eee;">${odds}+</td>
                            </tr>
                        `;
                    };
                    
                    const over15 = (pred.over_15_prob * 100).toFixed(1);
                    const over25 = (pred.over_25_prob * 100).toFixed(1);
                    const over35 = (pred.over_35_prob * 100).toFixed(1);
                    const over45 = (pred.over_45_prob * 100).toFixed(1);
                    const cornersOver = (pred.corners_over_prob * 100).toFixed(1);
                    const cardsOver = (pred.cards_over_prob * 100).toFixed(1);
                    const cornersOdds = odds.corners_over;
                    const cardsOdds = odds.cards_over;
                    
                    const goalsHTML = `
                        <div style="margin-top: 20px; text-align: left;">
                            
                            <h4 style="margin: 0 0 10px 0; color: #555; border-bottom: 2px solid #eee; padding-bottom: 5px;">⚽ Goals</h4>
                            <table style="width: 100%; border-collapse: collapse; font-size: 0.9em; margin-bottom: 20px;">
                                <thead>
                                    <tr style="background: #f8f9fa; color: #666;">
                                        <th style="padding:8px; text-align:left;">Market</th>
                                        <th style="padding:8px; text-align:left;">Probability</th>
                                        <th style="padding:8px; text-align:left;">Target Odds</th>
                                    </tr>
                                </thead>
                                <tbody>
                                    ${goalRow("Over 1.5 Goals", over15, odds.over_15)}
                                    ${goalRow("Over 2.5 Goals", over25, odds.over_25)}
                                    ${goalRow("Over 3.5 Goals", over35, odds.over_35)}
                                    ${goalRow("Over 4.5 Goals", over45, odds.over_45)}
                                </tbody>
                            </table>

                            <h4 style="margin: 0 0 10px 0; color: #555; border-bottom: 2px solid #eee; padding-bottom: 5px;">🚩 Corners</h4>
                            <table style="width: 100%; border-collapse: collapse; font-size: 0.9em; margin-bottom: 20px;">
                                <thead>
                                    <tr style="background: #f8f9fa; color: #666;">
                                        <th style="padding:8px; text-align:left;">Market</th>
                                        <th style="padding:8px; text-align:left;">Probability</th>
                                        <th style="padding:8px; text-align:left;">Target Odds</th>
                                    </tr>
                                </thead>
                                <tbody>
                                    ${goalRow("Over 9.5 Corners", cornersOver, cornersOdds)}
                                </tbody>
                            </table>

                            <h4 style="margin: 0 0 10px 0; color: #555; border-bottom: 2px solid #eee; padding-bottom: 5px;">🟨 Cards</h4>
                            <table style="width: 100%; border-collapse: collapse; font-size: 0.9em;">
                                <thead>
                                    <tr style="background: #f8f9fa; color: #666;">
                                        <th style="padding:8px; text-align:left;">Market</th>
                                        <th style="padding:8px; text-align:left;">Probability</th>
                                        <th style="padding:8px; text-align:left;">Target Odds</th>
                                    </tr>
                                </thead>
                                <tbody>
                                    ${goalRow("Over 3.5 Cards", cardsOver, cardsOdds)}
                                </tbody>
                            </table>
                        </div>
                    `;

                    const renderBadges = (badges) => {
                        if (!badges || badges.length === 0) return '';
                        return badges.map(b => `<span class="team-badge" style="background-color: ${b.color}">${b.text}</span>`).join('');
                    };

                    // Helper for history
                    const renderHistory = (matches) => matches.map(m => 
                        `<div class="match-row">
                            <span><span class="res-badge res-${m.result}">${m.result}</span> ${m.date}</span>
                            <span>vs ${m.opponent} <b>${m.score}</b></span>
                        </div>`
                    ).join('');
                    
                    const renderH2H = (matches) => matches.map(m =>
                        `<div class="match-row">
                            <span>${m.date}</span>
                            <span>${m.home} <b>${m.score}</b> ${m.away}</span>
                        </div>`
                    ).join('');
                    
                    const renderSimilarSplit = (homeMatches, awayMatches, homeName, awayName) => {
                        const renderList = (matches, title) => {
                            let h = `<div style="background:white; padding:10px; border-radius:8px; border:1px solid #eee; height: 100%;">`;
                            h += `<h4 style="margin:0 0 10px 0; font-size:0.9em; color:#666;">${title}</h4>`;
                            if (!matches || matches.length === 0) {
                                h += '<div style="color:#999; font-style:italic;">No matches found.</div>';
                            } else {
                                matches.forEach(m => {
                                    let badgeClass = "res-" + m.result;
                                    h += `<div class="match-row" style="font-size:0.78em;">
                                            <span style="white-space:nowrap"><span class="res-badge ${badgeClass}">${m.result}</span> <span style="color:#888; margin-left:4px;">${m.date}</span></span>
                                            <span style="white-space:nowrap; overflow:hidden; text-overflow:ellipsis; max-width:65%; text-align:right;">vs ${m.opponent} <span style="color:#999;">(${m.elo})</span> <b>${m.score}</b></span>
                                        </div>`;
                                });
                            }
                            h += '</div>';
                            return h;
                        };
                        return `<div class="history-container" style="margin-top:0;">
                                    ${renderList(homeMatches, `${homeName} vs Opponents like ${awayName}`)}
                                    ${renderList(awayMatches, `${awayName} vs Opponents like ${homeName}`)}
                                </div>`;
                    };

                resultDiv.innerHTML = `
                    <div class="animate-in">
                        ${rivalryHTML}
                        ${bestBetHTML}
                        
                        <h2>Match Outcome Probability</h2>
                        <div class="prob-grid">
                            <div class="prob-box">
                                <div>🏠 ${home}</div>
                                <div>${renderBadges(data.home_badges)}</div>
                                <strong style="display:block; margin-top:5px;">${data.home_win}%</strong>
                            </div>
                            <div class="prob-box"><div>🤝 Draw</div><strong>${data.draw}%</strong></div>
                            <div class="prob-box">
                                <div>✈️ ${away}</div>
                                <div>${renderBadges(data.away_badges)}</div>
                                <strong style="display:block; margin-top:5px;">${data.away_win}%</strong>
                            </div>
                        </div>

                        <button class="collapsible" onclick="toggleCollapsible(this)">📊 Stats & Analysis</button>
                        <div class="content">
                             <div class="analysis" style="border:none; margin-top:0;">
                                <br>
                                <div class="comp-row"><div class="comp-val home">${data.features_used.home_form_score}</div><div class="comp-lbl">Recent Form</div><div class="comp-val away">${data.features_used.away_form_score}</div></div>
                                <div class="comp-row"><div class="comp-val home">${Math.round(data.features_used.home_elo)}</div><div class="comp-lbl">ELO Rating</div><div class="comp-val away">${Math.round(data.features_used.away_elo)}</div></div>
                                <div class="comp-row"><div class="comp-val home">${data.features_used.home_home_form}</div><div class="comp-lbl">Venue Form</div><div class="comp-val away">${data.features_used.away_away_form}</div></div>
                                <div class="comp-row"><div class="comp-val home">${data.features_used.home_goals_scored}</div><div class="comp-lbl">Avg Goals Scored</div><div class="comp-val away">${data.features_used.away_goals_scored}</div></div>
                                <div class="comp-row"><div class="comp-val home">${data.features_used.home_goals_conceded}</div><div class="comp-lbl">Avg Goals Conceded</div><div class="comp-val away">${data.features_used.away_goals_conceded}</div></div>
                                <div class="comp-row"><div class="comp-val home">${data.features_used.home_shots_ot}</div><div class="comp-lbl">Avg Shots on Target</div><div class="comp-val away">${data.features_used.away_shots_ot}</div></div>
                                <div class="comp-row"><div class="comp-val home">${data.features_used.home_corners}</div><div class="comp-lbl">Avg Corners</div><div class="comp-val away">${data.features_used.away_corners}</div></div>
                                <div class="comp-row"><div class="comp-val home">${data.features_used.home_cards}</div><div class="comp-lbl">Avg Cards</div><div class="comp-val away">${data.features_used.away_cards}</div></div>
                                <div class="comp-row"><div class="comp-val home">${data.features_used.home_rest_days}d</div><div class="comp-lbl">Rest Days</div><div class="comp-val away">${data.features_used.away_rest_days}d</div></div>
                            </div>
                        </div>

                        <button class="collapsible" onclick="toggleCollapsible(this)">📈 Over/Under Markets</button>
                        <div class="content">${goalsHTML}</div>
                        
                        <button class="collapsible" onclick="toggleCollapsible(this)">⚔️ Head-to-Head</button>
                        <div class="content" style="padding-top:15px;">${data.h2h.length ? renderH2H(data.h2h) : 'No recent history found.'}</div>

                        <button class="collapsible" onclick="toggleCollapsible(this)">🔍 Performance vs Similar Opponents</button>
                        <div class="content" style="padding-top:15px;">${renderSimilarSplit(data.home_similar, data.away_similar, home, away)}</div>

                        <button class="collapsible" onclick="toggleCollapsible(this)">📅 Recent History</button>
                        <div class="content" style="padding-top:15px;">
                            <div class="history-container">
                                <div class="history-box">
                                    ${renderTeamStanding(data.home_standings, home, data.home_league_name)}
                                    <h3>${home} Last 5</h3>
                                    ${renderHistory(data.home_history)}
                                </div>
                                <div class="history-box">
                                    ${renderTeamStanding(data.away_standings, away, data.away_league_name)}
                                    <h3>${away} Last 5</h3>
                                    ${renderHistory(data.away_history)}
                                </div>
                            </div>
                        </div>
                    </div>
                `;
                } catch (e) {
                    resultDiv.innerHTML = `<div style="color:red">Connection Error: ${e}</div>`;
                }
            }
            
            async function scanMarket() {
                const resultDiv = document.getElementById('result');
                resultDiv.style.display = 'block';
                resultDiv.innerHTML = "Scanning upcoming matches for value... (This may take a few seconds)";
                
                try {
                    const response = await fetch('/scan');
                    const data = await response.json();

                    if (data.message) {
                        resultDiv.innerHTML = `<h2>🔍 Top Value Picks</h2><p style='color:#666; font-style:italic;'>${data.message}</p>`;
                        return;
                    }

                    const picks = data.picks;
                    const acca = data.acca;
                    
                    let html = '<h2>🔍 Top Value Picks</h2>';
                    if (picks.length === 0) html += '<p>No high-confidence bets found for the next 7 days.</p>';
                    
                    picks.forEach(pick => {
                        html += `<div style="background:white; padding:15px; border-radius:8px; border-left:5px solid #6f42c1; margin-bottom:10px; box-shadow:0 2px 5px rgba(0,0,0,0.05); text-align:left;">
                            <div style="font-size:0.9em; color:#666;">${pick.date} • ${pick.league}</div>
                            <div style="font-weight:bold; font-size:1.1em; margin:5px 0;">${pick.match}</div>
                            <div style="display:flex; justify-content:space-between; align-items:center;">
                                <span style="background:#e8f0fe; color:#1a73e8; padding:4px 8px; border-radius:4px; font-weight:bold;">${pick.market}</span>
                                <span>Prob: <b>${pick.prob}%</b> <span style="color:#ccc">|</span> Fair Odds: <b>${pick.fair_odds}</b></span>
                            </div>
                        </div>`;
                    });
                    
                    // Accumulator HTML
                    let accaHTML = '';
                    if (acca && acca.legs.length > 1) {
                        accaHTML = `
                            <div style="background: linear-gradient(135deg, #FFD700 0%, #FDB931 100%); color: #333; padding: 20px; border-radius: 12px; margin-bottom: 30px; box-shadow: 0 10px 20px rgba(253, 185, 49, 0.3); position: relative; overflow: hidden;">
                                <div style="position: absolute; top: -10px; right: -10px; font-size: 5em; opacity: 0.1;">🚀</div>
                                <h3 style="margin-top:0; border-bottom: 2px solid rgba(0,0,0,0.1); padding-bottom: 10px;">🚀 The Weekend Accumulator</h3>
                                <div style="background: rgba(255,255,255,0.6); border-radius: 8px; padding: 10px; margin-bottom: 15px;">
                        `;
                        
                        acca.legs.forEach(leg => {
                            accaHTML += `
                                <div style="display: flex; justify-content: space-between; border-bottom: 1px dashed rgba(0,0,0,0.2); padding: 8px 0;">
                                    <span>${leg.match} <span style="font-size: 0.8em; font-weight: bold;">(${leg.market})</span></span>
                                    <strong>${leg.fair_odds}</strong>
                                </div>
                            `;
                        });
                        
                        accaHTML += `
                                </div>
                                <div style="display: flex; justify-content: space-between; align-items: center; font-size: 1.2em;">
                                    <span>Combined Fair Odds:</span>
                                    <span style="font-size: 1.5em; font-weight: bold;">${acca.total_odds}</span>
                                </div>
                                <div style="text-align: right; font-size: 0.9em; margin-top: 5px;">$10 Bet Returns: <b>$${(acca.total_odds * 10).toFixed(2)}</b></div>
                            </div>
                        `;
                    }
                    
                    resultDiv.innerHTML = accaHTML + html;
                } catch (e) {
                    resultDiv.innerHTML = `<div style="color:red">Scan Error: ${e}</div>`;
                }
            }
        </script>
    </body>
    </html>
    """


@app.get("/report-image")
async def get_report_image(strategy: str):
    """
    Serves the generated backtesting report images.
    """
    path = settings.DATA_DIR / f"report_{strategy}.png"
    if not path.exists():
        raise HTTPException(status_code=404, detail="Report image not found. Please run the evaluation script first.")
    return FileResponse(path)


@app.get("/teams")
async def get_available_teams():
    """
    List all teams available in the dataset.
    """
    df = data_loader.load_data()
    if df.empty:
        raise HTTPException(status_code=503, detail="Failed to load data from Football-Data.co.uk. Check server logs.")

    all_teams = sorted(list(set(df['HomeTeam'].unique()) | set(df['AwayTeam'].unique())))
    return {"teams": all_teams}


@app.get("/predict")
async def predict_api(home: str, away: str):
    if training_status["state"] == "failed":
        raise HTTPException(status_code=500, detail=f"Model training failed: {training_status['error']}")

    if not predictor.is_trained:
        raise HTTPException(status_code=503, detail="Model is still training (approx 2 mins). Please refresh shortly.")

    # 1. Load Data
    df = data_loader.load_data()
    if df.empty:
        raise HTTPException(status_code=503, detail="Failed to load data.")

    # 2. Validate Teams
    all_teams = set(df['HomeTeam'].unique()) | set(df['AwayTeam'].unique())
    if home not in all_teams:
        raise HTTPException(status_code=404, detail=f"Team '{home}' not found.")
    if away not in all_teams:
        raise HTTPException(status_code=404, detail=f"Team '{away}' not found.")

    # Check for rivalry
    rivalry = get_rivalry_info(home, away)

    # 3. ELO & Features
    df, elo_dict = feature_engineer.enrich_with_elo(df)
    home_elo = elo_dict.get(home, 1500)
    away_elo = elo_dict.get(away, 1500)

    current_date = pd.Timestamp.now()
    features = feature_engineer.prepare_features(df, home, away, current_date, home_elo, away_elo)

    # 4. Predict
    pred = predictor.predict(features)

    # 5. Context Data
    home_history = get_recent_matches(df, home)
    away_history = get_recent_matches(df, away)
    h2h = get_h2h_history(df, home, away)
    
    # Badges
    home_badges = generate_badges(features, is_home=True)
    away_badges = generate_badges(features, is_home=False)

    # Similar Matches
    home_similar = get_matches_against_similar_elo(df, home, away_elo)
    away_similar = get_matches_against_similar_elo(df, away, home_elo)

    # Standings
    def get_latest_division(team_name):
        team_matches = df[(df['HomeTeam'] == team_name) | (df['AwayTeam'] == team_name)]
        if not team_matches.empty:
            return team_matches.iloc[-1]['Div']
        return None

    home_div = get_latest_division(home)
    away_div = get_latest_division(away)

    home_standings = get_standings(df, home_div) if home_div else []
    home_league_name = LEAGUE_NAMES.get(home_div, "Unknown League")

    if home_standings:
        for t in home_standings:
            if t['Team'] == home: t['RestDays'] = features[15]  # home_rest_days

    if away_div != home_div:
        away_standings = get_standings(df, away_div) if away_div else []
        away_league_name = LEAGUE_NAMES.get(away_div, "Unknown League")
    else:
        away_standings = home_standings
        away_league_name = home_league_name

    if away_standings:
        for t in away_standings:
            if t['Team'] == away: t['RestDays'] = features[16]  # away_rest_days

    # 6. Calculate Fair Odds & Best Bet
    def to_odds(prob):
        return round(1 / prob, 2) if prob > 0.01 else 99.00

    probs = {
        "Home Win": pred["home_win_prob"],
        "Away Win": pred["away_win_prob"],
        "Over 1.5 Goals": pred["over_15_prob"],
        "Over 2.5 Goals": pred["over_25_prob"],
        "Under 2.5 Goals": 1.0 - pred["over_25_prob"],
        "Over 9.5 Corners": pred["corners_over_prob"],
        "Over 3.5 Cards": pred["cards_over_prob"]
    }

    best_market = max(probs, key=probs.get)
    best_prob = probs[best_market]

    recommendation = None
    if best_prob > 0.60:
        recommendation = {
            "market": best_market,
            "probability": best_prob,
            "fair_odds": to_odds(best_prob)
        }

    response_data = {
        "home_win": round(pred["home_win_prob"] * 100, 1),
        "draw": round(pred["draw_prob"] * 100, 1),
        "away_win": round(pred["away_win_prob"] * 100, 1),
        "elo_gap": int(home_elo - away_elo),
        "rivalry": rivalry,
        "features_used": {
            "home_form_score": features[0],
            "away_form_score": features[1],
            "home_home_form": features[2],
            "away_away_form": features[3],
            "h2h_home_wins": features[4],
            "home_goals_scored": features[5],
            "away_goals_scored": features[6],
            "home_goals_conceded": features[7],
            "away_goals_conceded": features[8],
            "home_elo": features[9],
            "away_elo": features[10],
            "home_shots": features[11],
            "away_shots": features[12],
            "home_shots_ot": features[13],
            "away_shots_ot": features[14],
            "home_rest_days": features[15],
            "away_rest_days": features[16],
            "home_corners": features[17],
            "away_corners": features[18],
            "home_cards": features[19],
            "away_cards": features[20]
        },
        "prediction": pred,
        "home_badges": home_badges,
        "away_badges": away_badges,
        "recommendation": recommendation,
        "fair_odds": {
            "home_win": to_odds(pred["home_win_prob"]),
            "draw": to_odds(pred["draw_prob"]),
            "away_win": to_odds(pred["away_win_prob"]),
            "over_15": to_odds(pred["over_15_prob"]),
            "over_25": to_odds(pred["over_25_prob"]),
            "over_35": to_odds(pred["over_35_prob"]),
            "over_45": to_odds(pred["over_45_prob"]),
            "corners_over": to_odds(pred["corners_over_prob"]),
            "cards_over": to_odds(pred["cards_over_prob"])
        },
        "home_history": home_history,
        "away_history": away_history,
        "h2h": h2h,
        "home_similar": home_similar,
        "away_similar": away_similar,
        "home_standings": home_standings,
        "away_standings": away_standings,
        "home_league_name": home_league_name,
        "away_league_name": away_league_name
    }
    
    return sanitize_payload(response_data)


@app.get("/scan")
async def scan_market():
    """
    Scans upcoming fixtures and returns high-confidence predictions.
    """
    if training_status["state"] == "failed":
        return {"picks": [], "acca": None, "message": f"System Error: {training_status['error']}"}

    if not predictor.is_trained:
        return {"picks": [], "acca": None,
                "message": "Model is still training (approx 2 mins). Please try again shortly."}

    df = data_loader.load_data()
    if df.empty: return {"picks": [], "acca": None, "message": "Data could not be loaded."}

    # Filter for upcoming matches (where FTHG is NaN)
    today = pd.Timestamp.now().normalize()
    next_week = today + timedelta(days=7)
    upcoming = df[(df['Date'] >= today) & (df['Date'] <= next_week) & (df['FTHG'].isna())]

    picks = []

    # Pre-calc ELO
    df_hist, elo_dict = feature_engineer.enrich_with_elo(df)

    for _, row in upcoming.iterrows():
        home, away = row['HomeTeam'], row['AwayTeam']

        # Skip if we don't have enough history for these teams
        if home not in elo_dict or away not in elo_dict: continue

        home_elo = elo_dict[home]
        away_elo = elo_dict[away]

        # FIX: Ensure we only use data from before the match to calculate features
        historical_data_for_match = df_hist[df_hist['Date'] < row['Date']]
        features = feature_engineer.prepare_features(historical_data_for_match, home, away, row['Date'], home_elo,
                                                     away_elo)
        pred = predictor.predict(features)

        # Check for high confidence (>65%)
        # We check Over 2.5 Goals specifically as it was our best performing model
        prob = pred["over_25_prob"]
        if prob > 0.65:
            picks.append({
                "date": row['Date'].strftime("%a %d %b"),
                "league": LEAGUE_NAMES.get(row['Div'], row['Div']),
                "match": f"{home} vs {away}",
                "market": "Over 2.5 Goals",
                "prob": round(prob * 100, 1),
                "fair_odds": round(1 / prob, 2)
            })

    # Sort by probability (highest first)
    picks.sort(key=lambda x: x['prob'], reverse=True)

    # Generate Accumulator (Top 3 safest bets)
    # Filter for very high confidence (>70%) to be safe
    safe_bets = [p for p in picks if p['prob'] > 70]

    # Take top 3, or top 3 from picks if not enough safe ones
    acca_legs = safe_bets[:3] if len(safe_bets) >= 3 else picks[:3]

    acca_odds = 1.0
    for leg in acca_legs:
        acca_odds *= leg['fair_odds']

    acca = {
        "legs": acca_legs,
        "total_odds": round(acca_odds, 2)
    }

    return sanitize_payload({
        "picks": picks[:10],
        "acca": acca
    })
