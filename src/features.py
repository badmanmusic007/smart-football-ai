import pandas as pd
import numpy as np
from src.weather import get_weather_forecast


class FeatureEngineer:

    @staticmethod
    def calculate_form(df, team_name, last_n=5):
        """
        Calculate form points from the last N matches in the dataframe.
        Win = 3 pts, Draw = 1 pt, Loss = 0 pts.
        """
        # Filter for played matches and then for the team
        played_df = df.dropna(subset=['FTR'])
        team_matches = played_df[(played_df['HomeTeam'] == team_name) | (played_df['AwayTeam'] == team_name)].tail(
            last_n)

        points = 0
        for _, match in team_matches.iterrows():
            if match['HomeTeam'] == team_name:
                if match['FTHG'] > match['FTAG']:
                    points += 3
                elif match['FTHG'] == match['FTAG']:
                    points += 1
            else:  # Away team
                if match['FTAG'] > match['FTHG']:
                    points += 3
                elif match['FTAG'] == match['FTHG']:
                    points += 1

        return points

    def calculate_specific_form(self, df, team_name, is_home, last_n=5):
        """
        Calculate form points specifically for Home or Away matches.
        """
        played_df = df.dropna(subset=['FTR'])
        if is_home:
            matches = played_df[played_df['HomeTeam'] == team_name].tail(last_n)
        else:
            matches = played_df[played_df['AwayTeam'] == team_name].tail(last_n)

        points = 0
        for _, match in matches.iterrows():
            if is_home:
                if match['FTHG'] > match['FTAG']:
                    points += 3
                elif match['FTHG'] == match['FTAG']:
                    points += 1
            else:
                if match['FTAG'] > match['FTHG']:
                    points += 3
                elif match['FTAG'] == match['FTHG']:
                    points += 1
        return points

    @staticmethod
    def calculate_goal_stats(df, team_name, last_n=5):
        """
        Calculate average goals scored and conceded in last N matches using EWMA.
        """
        played_df = df.dropna(subset=['FTR'])
        team_matches = played_df[(played_df['HomeTeam'] == team_name) | (played_df['AwayTeam'] == team_name)]

        if len(team_matches) < last_n:
            return 0, 0

        # Create series for goals scored and conceded by the team
        scored = pd.Series(np.where(team_matches['HomeTeam'] == team_name, team_matches['FTHG'], team_matches['FTAG']),
                           index=team_matches.index)
        conceded = pd.Series(
            np.where(team_matches['HomeTeam'] == team_name, team_matches['FTAG'], team_matches['FTHG']),
            index=team_matches.index)

        # Calculate the exponentially weighted moving average
        avg_scored = scored.ewm(span=last_n, adjust=False).mean().iloc[-1]
        avg_conceded = conceded.ewm(span=last_n, adjust=False).mean().iloc[-1]

        return round(avg_scored, 2), round(avg_conceded, 2)

    @staticmethod
    def calculate_shot_stats(df, team_name, last_n=5):
        """
        Calculate average Shots and Shots on Target in last N matches using EWMA.
        """
        played_df = df.dropna(subset=['FTR'])
        team_matches = played_df[(played_df['HomeTeam'] == team_name) | (played_df['AwayTeam'] == team_name)]

        if len(team_matches) < last_n:
            return 0, 0

        # Create series for shots and shots on target
        shots = pd.Series(np.where(team_matches['HomeTeam'] == team_name, team_matches['HS'], team_matches['AS']),
                          index=team_matches.index)
        shots_ot = pd.Series(np.where(team_matches['HomeTeam'] == team_name, team_matches['HST'], team_matches['AST']),
                             index=team_matches.index)

        avg_shots = shots.ewm(span=last_n, adjust=False).mean().iloc[-1]
        avg_shots_ot = shots_ot.ewm(span=last_n, adjust=False).mean().iloc[-1]

        return round(avg_shots, 2), round(avg_shots_ot, 2)

    @staticmethod
    def calculate_conceded_shot_stats(df, team_name, last_n=5):
        """
        Calculate average Shots on Target Conceded in last N matches using EWMA.
        """
        played_df = df.dropna(subset=['FTR'])
        team_matches = played_df[(played_df['HomeTeam'] == team_name) | (played_df['AwayTeam'] == team_name)]

        if len(team_matches) < last_n:
            return 0

        # Inverse of shots taken (Opponent's shots on target)
        shots_ot_conceded = pd.Series(
            np.where(team_matches['HomeTeam'] == team_name, team_matches['AST'], team_matches['HST']),
            index=team_matches.index)

        avg_shots_ot_conceded = shots_ot_conceded.ewm(span=last_n, adjust=False).mean().iloc[-1]
        return round(avg_shots_ot_conceded, 2)

    @staticmethod
    def calculate_corner_stats(df, team_name, last_n=5):
        """
        Calculate average Corners in last N matches using EWMA.
        """
        played_df = df.dropna(subset=['FTR'])
        team_matches = played_df[(played_df['HomeTeam'] == team_name) | (played_df['AwayTeam'] == team_name)]

        if len(team_matches) < last_n:
            return 0

        corners = pd.Series(np.where(team_matches['HomeTeam'] == team_name, team_matches['HC'], team_matches['AC']),
                            index=team_matches.index)

        avg_corners = corners.ewm(span=last_n, adjust=False).mean().iloc[-1]
        return round(avg_corners, 2)

    @staticmethod
    def calculate_card_stats(df, team_name, last_n=5):
        """
        Calculate average Cards (Yellow + Red) in last N matches using EWMA.
        """
        played_df = df.dropna(subset=['FTR'])
        team_matches = played_df[(played_df['HomeTeam'] == team_name) | (played_df['AwayTeam'] == team_name)]

        if len(team_matches) < last_n:
            return 0

        cards = pd.Series(np.where(team_matches['HomeTeam'] == team_name, team_matches['HY'] + team_matches['HR'],
                                   team_matches['AY'] + team_matches['AR']), index=team_matches.index)

        avg_cards = cards.ewm(span=last_n, adjust=False).mean().iloc[-1]
        return round(avg_cards, 2)

    @staticmethod
    def calculate_rest_days(df, team_name, current_match_date):
        """
        Calculate days since the last match for a team.
        """
        # Filter for played matches for the team BEFORE the current date
        team_matches = df[
            ((df['HomeTeam'] == team_name) | (df['AwayTeam'] == team_name)) & (df['Date'] < current_match_date) & (
                df['FTR'].notna())]

        if team_matches.empty:
            return 7  # Default to 7 days rest if no history

        last_match_date = team_matches['Date'].max()
        delta = (current_match_date - last_match_date).days
        return min(delta, 14)  # Cap at 14 days (fully rested)

    @staticmethod
    def calculate_sos(df, team_name, last_n=5):
        """
        Calculate Strength of Schedule (Average Opponent ELO in last N matches).
        """
        # Filter for played matches involving the team
        played_df = df.dropna(subset=['FTR'])
        team_matches = played_df[(played_df['HomeTeam'] == team_name) | (played_df['AwayTeam'] == team_name)].tail(
            last_n)

        if team_matches.empty:
            return 1500  # Default average

        opponent_elos = []
        for _, row in team_matches.iterrows():
            opp_elo = row['AwayElo'] if row['HomeTeam'] == team_name else row['HomeElo']
            opponent_elos.append(opp_elo)

        return int(sum(opponent_elos) / len(opponent_elos))

    @staticmethod
    def calculate_deep_dive(df, team_name):
        """
        Calculate deep dive stats for the current season.
        Dynamically determines the season start date based on the latest match date in the dataframe.
        """
        if df.empty:
            return None
            
        # Determine the season start based on the latest date in the dataframe
        # Assuming seasons start in July
        latest_date = df['Date'].max()
        if latest_date.month >= 7:
            season_start = pd.Timestamp(year=latest_date.year, month=7, day=1)
        else:
            season_start = pd.Timestamp(year=latest_date.year - 1, month=7, day=1)

        # Filter for played matches in current season involving the team
        matches = df[
            ((df['HomeTeam'] == team_name) | (df['AwayTeam'] == team_name)) &
            (df['Date'] > season_start) &
            (df['FTR'].notna())
            ].copy()

        if matches.empty:
            return None

        stats = {
            "matches_played": len(matches),
            "total_shots": 0,
            "total_sot": 0,
            "goals_from_sot": 0,
            "leading_at_ht": 0,
            "won_when_leading_ht": 0,
            "losing_at_ht": 0,
            "won_when_losing_ht": 0,  # Comebacks
            "clean_sheets": 0,
            "failed_to_score": 0
        }

        for _, row in matches.iterrows():
            is_home = row['HomeTeam'] == team_name

            # Shooting
            shots = row['HS'] if is_home else row['AS']
            sot = row['HST'] if is_home else row['AST']
            goals = row['FTHG'] if is_home else row['FTAG']
            opponent_goals = row['FTAG'] if is_home else row['FTHG']

            stats["total_shots"] += shots
            stats["total_sot"] += sot
            stats["goals_from_sot"] += goals

            # Consistency
            if opponent_goals == 0: stats["clean_sheets"] += 1
            if goals == 0: stats["failed_to_score"] += 1

            # Game State (HT/FT)
            ht_home = row['HTHG']
            ht_away = row['HTAG']

            # Check if leading at HT
            leading_ht = (is_home and ht_home > ht_away) or (not is_home and ht_away > ht_home)
            losing_ht = (is_home and ht_home < ht_away) or (not is_home and ht_away < ht_home)
            won_ft = (is_home and row['FTR'] == 'H') or (not is_home and row['FTR'] == 'A')

            if leading_ht:
                stats["leading_at_ht"] += 1
                if won_ft: stats["won_when_leading_ht"] += 1

            if losing_ht:
                stats["losing_at_ht"] += 1
                if won_ft: stats["won_when_losing_ht"] += 1

        return stats

    def enrich_with_elo(self, df):
        """
        Iterate through the dataframe and calculate ELO ratings for every match.
        Returns the dataframe with 'HomeElo' and 'AwayElo' columns, plus the final ratings dict.
        """
        elo_dict = {}  # Stores current rating for each team
        k_factor = 20  # How much ratings change per match

        home_elos = []
        away_elos = []

        for _, row in df.iterrows():
            h_team = row['HomeTeam']
            a_team = row['AwayTeam']

            # Get current ratings (default 1500)
            h_elo = elo_dict.get(h_team, 1500)
            a_elo = elo_dict.get(a_team, 1500)

            home_elos.append(h_elo)
            away_elos.append(a_elo)

            # Calculate Expected Result
            # E = 1 / (1 + 10 ^ ((OppElo - OwnElo) / 400))
            h_expected = 1 / (1 + 10 ** ((a_elo - h_elo) / 400))
            a_expected = 1 / (1 + 10 ** ((h_elo - a_elo) / 400))

            # Actual Result (1=Win, 0.5=Draw, 0=Loss)
            if pd.isna(row['FTR']):
                h_actual, a_actual = None, None
            elif row['FTR'] == 'H':
                h_actual, a_actual = 1, 0
            elif row['FTR'] == 'D':
                h_actual, a_actual = 0.5, 0.5
            else:
                h_actual, a_actual = 0, 1

            # Update Ratings
            if h_actual is not None:
                elo_dict[h_team] = h_elo + k_factor * (h_actual - h_expected)
                elo_dict[a_team] = a_elo + k_factor * (a_actual - a_expected)

        df['HomeElo'] = home_elos
        df['AwayElo'] = away_elos
        return df, elo_dict

    def prepare_features(self, df, home_team, away_team, match_date, division, referee, home_elo=1500, away_elo=1500, fetch_weather=True):
        """
        Generate features for a specific matchup using historical data.
        """
        # 1. Form (Last 5 games)
        home_form = self.calculate_form(df, home_team)
        away_form = self.calculate_form(df, away_team)

        # 1b. Specific Form (Home vs Away)
        home_home_form = self.calculate_specific_form(df, home_team, is_home=True)
        away_away_form = self.calculate_specific_form(df, away_team, is_home=False)

        # 2. Head-to-Head (H2H) - Home Wins in history
        h2h_matches = df[
            ((df['HomeTeam'] == home_team) & (df['AwayTeam'] == away_team))
        ]

        h2h_home_wins = 0
        for _, match in h2h_matches.iterrows():
            if match['FTHG'] > match['FTAG']:
                h2h_home_wins += 1

        # 3. Goals (Attack/Defense)
        home_scored, home_conceded = self.calculate_goal_stats(df, home_team)
        away_scored, away_conceded = self.calculate_goal_stats(df, away_team)

        # 4. Shot Stats (Performance)
        home_shots, home_shots_ot = self.calculate_shot_stats(df, home_team)
        away_shots, away_shots_ot = self.calculate_shot_stats(df, away_team)

        # 5. Fatigue (Rest Days)
        home_rest = self.calculate_rest_days(df, home_team, match_date)
        away_rest = self.calculate_rest_days(df, away_team, match_date)

        # 6. Corners
        home_corners = self.calculate_corner_stats(df, home_team)
        away_corners = self.calculate_corner_stats(df, away_team)

        # 7. Cards (Discipline)
        home_cards = self.calculate_card_stats(df, home_team)
        away_cards = self.calculate_card_stats(df, away_team)

        # 8. League Context (One-Hot Encoding)
        leagues = ['E0', 'E1', 'SP1', 'D1', 'I1', 'F1', 'N1', 'P1', 'SC0', 'E2', 'E3', 'SP2', 'D2', 'I2', 'F2', 'B1', 'T1', 'G1', 'SC1', 'CL', 'EL', 'ECL', 'FAC', 'LC', 'DFB', 'CDR', 'CDF', 'CI']
        league_features = [1 if division == l else 0 for l in leagues]

        # 9. Efficiency (Clinicality & Resilience)
        home_shots_ot_c = self.calculate_conceded_shot_stats(df, home_team)
        away_shots_ot_c = self.calculate_conceded_shot_stats(df, away_team)

        home_conversion = round(home_scored / home_shots_ot, 3) if home_shots_ot > 0 else 0.0
        away_conversion = round(away_scored / away_shots_ot, 3) if away_shots_ot > 0 else 0.0

        home_save_ratio = round(1 - (home_conceded / home_shots_ot_c), 3) if home_shots_ot_c > 0 else 0.5
        away_save_ratio = round(1 - (away_conceded / away_shots_ot_c), 3) if away_shots_ot_c > 0 else 0.5

        # 10. Strength of Schedule (SoS)
        home_sos = self.calculate_sos(df, home_team)
        away_sos = self.calculate_sos(df, away_team)
        
        # 11. Weather
        if fetch_weather:
            weather = get_weather_forecast(home_team, match_date)
            rain_mm = weather['rain_mm']
            temp_c = weather['temperature_c']
        else:
            # Use neutral defaults for historical matches
            rain_mm = 0.0
            temp_c = 10.0

        base_features = [
            home_form, away_form, home_home_form, away_away_form, h2h_home_wins,
            home_scored, away_scored, home_conceded, away_conceded,
            home_elo, away_elo,
            home_shots, away_shots, home_shots_ot, away_shots_ot,
            home_rest, away_rest,
            home_corners, away_corners,
            home_cards, away_cards
        ]

        advanced_features = [
            home_conversion, away_conversion,
            home_save_ratio, away_save_ratio,
            home_sos, away_sos,
            rain_mm, temp_c
        ]

        return base_features + advanced_features + league_features
