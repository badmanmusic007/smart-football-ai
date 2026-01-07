import pandas as pd
import logging
import urllib.request
import ssl
import os
import numpy as np
import time
from io import StringIO
from concurrent.futures import ThreadPoolExecutor, as_completed
from src.config import settings


logger = logging.getLogger(__name__)

class FootballDataLoader:
    def __init__(self):
        self.data = None
        self.cache_file = settings.DATA_DIR / "match_data_final_v24.csv"
        self.advanced_stats_file = settings.DATA_DIR / "advanced_stats.csv"

    def _download_single_csv(self, url):
        """Helper to download and process a single CSV URL."""
        headers = {
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        }
        
        # Create an unverified SSL context to bypass certificate errors
        ssl_context = ssl.create_default_context()
        ssl_context.check_hostname = False
        ssl_context.verify_mode = ssl.CERT_NONE
        
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, context=ssl_context) as response:
                data = response.read().decode('latin-1') # CSVs are often latin-1 encoded
            
            csv_data = StringIO(data)
            df = pd.read_csv(csv_data)
            
            # Force Div column from URL filename to ensure correct League separation
            # This fixes the issue where all teams were lumped into one league
            # EXCEPTION: fixtures.csv contains mixed leagues, so we trust its internal 'Div' column
            if 'fixtures.csv' not in url:
                div_code = url.split('/')[-1].replace('.csv', '')
                df['Div'] = div_code

            # Define columns we want
            # Map our internal names to potential CSV column names (Priority order)
            col_mapping = {
                'B365BTTSY': ['B365BTTSY', 'BbAvBTTSY', 'AvgBTTSY', 'BTTSY'],
                'B365BTTSN': ['B365BTTSN', 'BbAvBTTSN', 'AvgBTTSN', 'BTTSN'],
                'B365>2.5': ['B365>2.5', 'BbAv>2.5', 'Avg>2.5'],
                'B365<2.5': ['B365<2.5', 'BbAv<2.5', 'Avg<2.5'],
                'AvgH': ['AvgH', 'BbAvH'],
                'AvgD': ['AvgD', 'BbAvD'],
                'AvgA': ['AvgA', 'BbAvA'],
                'MaxH': ['MaxH', 'BbMxH'],
                'MaxD': ['MaxD', 'BbMxD'],
                'MaxA': ['MaxA', 'BbMxA']
            }
            
            # Normalize columns: If B365BTTSY is missing but BbAvBTTSY exists, rename it
            for target, candidates in col_mapping.items():
                if target not in df.columns:
                    for candidate in candidates:
                        if candidate in df.columns:
                            df[target] = df[candidate]
                            break
            
            expected_cols = [
                'Div', 'Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'FTR', 'HTHG', 'HTAG', 'Referee', 
                'B365H', 'B365D', 'B365A', 'B365>2.5', 'B365<2.5', 'B365BTTSY', 'B365BTTSN', 
                'HS', 'AS', 'HST', 'AST', 'HC', 'AC', 'HY', 'AY', 'HR', 'AR',
                'AvgH', 'AvgD', 'AvgA', 'MaxH', 'MaxD', 'MaxA'
            ]

            # Handle missing columns (e.g. BTTS odds might not exist in all leagues)
            for col in expected_cols:
                if col not in df.columns:
                    if col == 'Div': df[col] = 'Unknown'
                    elif col == 'Referee': df[col] = 'Unknown'
                    # Use np.nan for missing numerical data so isnull() works correctly
                    elif col in ['FTHG', 'FTAG', 'FTR', 'HTHG', 'HTAG']: df[col] = np.nan
                    # Fill stats/odds with 0.0 to prevent model crashes
                    else: df[col] = 0.0
            
            # Fill NaNs in existing stats/odds columns with 0.0
            stats_cols = [
                'HS', 'AS', 'HST', 'AST', 'HC', 'AC', 'HY', 'AY', 'HR', 'AR', 
                'B365H', 'B365D', 'B365A', 'B365>2.5', 'B365<2.5', 'B365BTTSY', 'B365BTTSN',
                'AvgH', 'AvgD', 'AvgA', 'MaxH', 'MaxD', 'MaxA'
            ]
            for col in stats_cols:
                if col in df.columns:
                    df[col] = df[col].fillna(0.0)
            
            # Keep only relevant columns and drop rows with missing core data
            # We keep rows even if FTHG is missing (upcoming matches), but ensure Date/Teams exist
            df = df[expected_cols].dropna(subset=['Date', 'HomeTeam', 'AwayTeam'])
            return df
        except Exception as e:
            logger.warning(f"Could not load data from {url}: {e}")
            return None

    def merge_advanced_stats(self, main_df):
        """Merges scraped advanced stats (xG, etc.) into the main dataframe."""
        if not self.advanced_stats_file.exists():
            logger.warning("Advanced stats file not found. Skipping merge.")
            return main_df

        try:
            adv_df = pd.read_csv(self.advanced_stats_file)
            
            # Create a mapping dictionary for O(1) lookup
            # Key: (Team, Div) -> Stats
            stats_map = {}
            for _, row in adv_df.iterrows():
                key = (row['Team'], row['Div'])
                stats_map[key] = {
                    'xG_per90': row.get('xG_per90', 0),
                    'xGA_per90': row.get('xGA_per90', 0),
                    'Possession': row.get('Possession', 50)
                }
            
            # Apply to main dataframe
            # We add columns for Home and Away teams
            home_xg, home_xga, home_poss = [], [], []
            away_xg, away_xga, away_poss = [], [], []
            
            for _, row in main_df.iterrows():
                h_key = (row['HomeTeam'], row['Div'])
                a_key = (row['AwayTeam'], row['Div'])
                
                h_stats = stats_map.get(h_key, {'xG_per90': 1.3, 'xGA_per90': 1.3, 'Possession': 50}) # Default values
                a_stats = stats_map.get(a_key, {'xG_per90': 1.3, 'xGA_per90': 1.3, 'Possession': 50})
                
                home_xg.append(h_stats['xG_per90'])
                home_xga.append(h_stats['xGA_per90'])
                home_poss.append(h_stats['Possession'])
                
                away_xg.append(a_stats['xG_per90'])
                away_xga.append(a_stats['xGA_per90'])
                away_poss.append(a_stats['Possession'])
                
            main_df['Home_xG'] = home_xg
            main_df['Home_xGA'] = home_xga
            main_df['Home_Poss'] = home_poss
            main_df['Away_xG'] = away_xg
            main_df['Away_xGA'] = away_xga
            main_df['Away_Poss'] = away_poss
            
            print("Advanced stats merged successfully.")
            return main_df
            
        except Exception as e:
            logger.error(f"Failed to merge advanced stats: {e}")
            return main_df

    def load_data(self):
        """
        Download and combine CSV data from Football-Data.co.uk
        """
        # 1. Check In-Memory Cache
        if self.data is not None:
            return self.data
            
        # 2. Check Disk Cache (Valid for 24 hours)
        if self.cache_file.exists():
            # Check if file is older than 24 hours (86400 seconds)
            if time.time() - os.path.getmtime(self.cache_file) < 86400:
                print("Loading data from local disk cache...")
                try:
                    df = pd.read_csv(self.cache_file)
                    df['Date'] = pd.to_datetime(df['Date'])
                    self.data = df.sort_values('Date')
                    return self.data
                except Exception as e:
                    logger.warning(f"Cache file corrupted, downloading fresh data: {e}")
            
        all_data = []
        print("Downloading match data from Football-Data.co.uk (Parallel)...")
        
        with ThreadPoolExecutor(max_workers=5) as executor:
            future_to_url = {executor.submit(self._download_single_csv, url): url for url in settings.DATA_URLS}
            for future in as_completed(future_to_url):
                df = future.result()
                if df is not None and not df.empty:
                    all_data.append(df)
            
        if all_data:
            combined_df = pd.concat(all_data, ignore_index=True)
            # Convert Date to datetime
            combined_df['Date'] = pd.to_datetime(combined_df['Date'], dayfirst=True)
            
            # Merge Advanced Stats
            combined_df = self.merge_advanced_stats(combined_df)

            self.data = combined_df.sort_values('Date')
            
            # Save to disk cache
            try:
                self.data.to_csv(self.cache_file, index=False)
                print("Data saved to local cache.")
            except Exception as e:
                logger.warning(f"Could not save cache: {e}")
                
            return self.data
        else:
            logger.error("No data could be loaded from any source.")
            return pd.DataFrame()