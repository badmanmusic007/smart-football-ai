import requests
import pandas as pd

# A simple mapping of teams to their stadium coordinates.
# This would ideally be in a more robust database.
STADIUM_COORDINATES = {
    "Arsenal": (51.555, -0.1086), "Aston Villa": (52.509, -1.8847),
    "Brentford": (51.4906, -0.2892), "Brighton": (50.861, -0.0832),
    "Burnley": (53.789, -2.2302), "Chelsea": (51.4817, -0.191),
    "Crystal Palace": (51.3983, -0.0856), "Everton": (53.4388, -2.9663),
    "Fulham": (51.475, -0.2217), "Leeds": (53.777, -1.572),
    "Leicester": (52.6204, -1.1422), "Liverpool": (53.4308, -2.9608),
    "Man City": (53.483, -2.2002), "Man United": (53.463, -2.2913),
    "Newcastle": (54.975, -1.621), "Norwich": (52.622, 1.309),
    "Southampton": (50.9058, -1.3911), "Tottenham": (51.604, -0.066),
    "Watford": (51.650, -0.401), "West Ham": (51.5386, -0.0166),
    "Wolves": (52.5902, -2.1303),
    # Spanish Teams
    "Real Madrid": (40.453, -3.688), "Barcelona": (41.380, 2.122),
    "Atletico Madrid": (40.436, -3.600), "Sevilla": (37.384, -5.970),
    # German Teams
    "Bayern Munich": (48.218, 11.624), "Dortmund": (51.492, 7.451),
    # Italian Teams
    "Juventus": (45.109, 7.641), "AC Milan": (45.478, 9.123), "Inter": (45.478, 9.123),
    # French Teams
    "Paris SG": (48.841, 2.253)
}

def get_weather_forecast(team_name: str, match_date: pd.Timestamp):
    """
    Get weather forecast for a given team's stadium on a specific date.
    """
    if team_name not in STADIUM_COORDINATES:
        return {"rain_mm": 0.0, "temperature_c": 10.0} # Return default if no coords

    lat, lon = STADIUM_COORDINATES[team_name]
    date_str = match_date.strftime('%Y-%m-%d')

    url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&daily=temperature_2m_mean,rain_sum&timezone=auto&start_date={date_str}&end_date={date_str}"

    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        
        if 'daily' in data and data['daily']['rain_sum'] and data['daily']['temperature_2m_mean']:
            return {
                "rain_mm": data['daily']['rain_sum'][0],
                "temperature_c": data['daily']['temperature_2m_mean'][0]
            }
    except requests.RequestException as e:
        print(f"Weather API error: {e}")
        
    # Return default on error
    return {"rain_mm": 0.0, "temperature_c": 10.0}
