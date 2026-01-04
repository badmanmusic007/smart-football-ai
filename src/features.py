import pandas as pd
import numpy as np

class FeatureEngineer:
    
    @staticmethod
    def calculate_form(df, team_name, last_n=5):
        """
        Calculate form points from the last N matches in the dataframe.
        Win = 3 pts, Draw = 1 pt, Loss = 0 pts.
        """
        # Filter for played matches and then for the team
        played_df = df.dropna(subset=['FTR'])
        team_matches = played_df[(played_df['HomeTeam'] == team_name) | (played_df['AwayTeam'] == team_name)].tail(last_n)
        
        points = 0
        for _, match in team_matches.iterrows():
            if match['HomeTeam'] == team_name:
                if match['FTHG'] > match['FTAG']: points += 3
                elif match['FTHG'] == match['FTAG']: points += 1
            else: # Away team
                if match['FTAG'] > match['FTHG']: points += 3
                elif match['FTAG'] == match['FTHG']: points += 1
                
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
                if match['FTHG'] > match['FTAG']: points += 3
                elif match['FTHG'] == match['FTAG']: points += 1
            else:
                if match['FTAG'] > match['FTHG']: points += 3
                elif match['FTAG'] == match['FTHG']: points += 1
        return points

    @staticmethod
    def calculate_goal_stats(df, team_name, last_n=5):
        """
        Calculate average goals scored and conceded in last N matches.
        """
        played_df = df.dropna(subset=['FTR'])
        team_matches = played_df[(played_df['HomeTeam'] == team_name) | (played_df['AwayTeam'] == team_name)].tail(last_n)
        
        if team_matches.empty:
            return 0, 0
            
        scored = 0
        conceded = 0
        
        for _, match in team_matches.iterrows():
            if match['HomeTeam'] == team_name:
                scored += match['FTHG']
                conceded += match['FTAG']
            else:
                scored += match['FTAG']
                conceded += match['FTHG']
                
        avg_scored = round(scored / len(team_matches), 2)
        avg_conceded = round(conceded / len(team_matches), 2)
        
        return avg_scored, avg_conceded

    @staticmethod
    def calculate_shot_stats(df, team_name, last_n=5):
        """
        Calculate average Shots and Shots on Target (HST/AST) in last N matches.
        """
        played_df = df.dropna(subset=['FTR'])
        team_matches = played_df[(played_df['HomeTeam'] == team_name) | (played_df['AwayTeam'] == team_name)].tail(last_n)
        
        if team_matches.empty:
            return 0, 0
            
        shots = 0
        shots_ot = 0
        
        for _, match in team_matches.iterrows():
            if match['HomeTeam'] == team_name:
                shots += match['HS']
                shots_ot += match['HST']
            else:
                shots += match['AS']
                shots_ot += match['AST']
                
        avg_shots = round(shots / len(team_matches), 2)
        avg_shots_ot = round(shots_ot / len(team_matches), 2)
        
        return avg_shots, avg_shots_ot

    @staticmethod
    def calculate_corner_stats(df, team_name, last_n=5):
        """
        Calculate average Corners in last N matches.
        """
        played_df = df.dropna(subset=['FTR'])
        team_matches = played_df[(played_df['HomeTeam'] == team_name) | (played_df['AwayTeam'] == team_name)].tail(last_n)
        
        if team_matches.empty:
            return 0
            
        corners = 0
        for _, match in team_matches.iterrows():
            if match['HomeTeam'] == team_name:
                corners += match['HC']
            else:
                corners += match['AC']
                
        avg_corners = round(corners / len(team_matches), 2)
        return avg_corners

    @staticmethod
    def calculate_card_stats(df, team_name, last_n=5):
        """
        Calculate average Cards (Yellow + Red) in last N matches.
        """
        played_df = df.dropna(subset=['FTR'])
        team_matches = played_df[(played_df['HomeTeam'] == team_name) | (played_df['AwayTeam'] == team_name)].tail(last_n)
        
        if team_matches.empty:
            return 0
            
        cards = 0
        for _, match in team_matches.iterrows():
            if match['HomeTeam'] == team_name:
                cards += match.get('HY', 0) + match.get('HR', 0)
            else:
                cards += match.get('AY', 0) + match.get('AR', 0)
                
        avg_cards = round(cards / len(team_matches), 2)
        return avg_cards

    @staticmethod
    def calculate_rest_days(df, team_name, current_match_date):
        """
        Calculate days since the last match for a team.
        """
        # Filter for played matches for the team BEFORE the current date
        team_matches = df[((df['HomeTeam'] == team_name) | (df['AwayTeam'] == team_name)) & (df['Date'] < current_match_date) & (df['FTR'].notna())]
        
        if team_matches.empty:
            return 7 # Default to 7 days rest if no history
            
        last_match_date = team_matches['Date'].max()
        delta = (current_match_date - last_match_date).days
        return min(delta, 14) # Cap at 14 days (fully rested)

    def enrich_with_elo(self, df):
        """
        Iterate through the dataframe and calculate ELO ratings for every match.
        Returns the dataframe with 'HomeElo' and 'AwayElo' columns, plus the final ratings dict.
        """
        elo_dict = {} # Stores current rating for each team
        k_factor = 20 # How much ratings change per match
        
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
            elif row['FTR'] == 'H': h_actual, a_actual = 1, 0
            elif row['FTR'] == 'D': h_actual, a_actual = 0.5, 0.5
            else: h_actual, a_actual = 0, 1
            
            # Update Ratings
            if h_actual is not None:
                elo_dict[h_team] = h_elo + k_factor * (h_actual - h_expected)
                elo_dict[a_team] = a_elo + k_factor * (a_actual - a_expected)
            
        df['HomeElo'] = home_elos
        df['AwayElo'] = away_elos
        return df, elo_dict

    def prepare_features(self, df, home_team, away_team, match_date, home_elo=1500, away_elo=1500):
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
        
        # 7. Cards (Discipline) - This was missing from your file
        home_cards = self.calculate_card_stats(df, home_team)
        away_cards = self.calculate_card_stats(df, away_team)
        
        # The list of features now needs to be updated to include cards
        # Assuming your config.py and train_model.py are from Phase 16, this will work.
        # If not, you might need to remove home_cards and away_cards from this return list.
        
        return [home_form, away_form, home_home_form, away_away_form, h2h_home_wins, home_scored, away_scored, home_conceded, away_conceded, home_elo, away_elo, home_shots, away_shots, home_shots_ot, away_shots_ot, home_rest, away_rest, home_corners, away_corners, home_cards, away_cards]