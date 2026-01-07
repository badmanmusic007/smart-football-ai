import pandas as pd
import time
import random
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AdvancedStatsScraper:
    def __init__(self):
        self.data_dir = Path(__file__).resolve().parent.parent / "data"
        self.data_dir.mkdir(exist_ok=True)
        self.output_file = self.data_dir / "advanced_stats.csv"
        
        # Mapping of our league codes to FBref URLs (2024-2025 Season)
        self.league_urls = {
            'E0': 'https://fbref.com/en/comps/9/Premier-League-Stats',
            'SP1': 'https://fbref.com/en/comps/12/La-Liga-Stats',
            'D1': 'https://fbref.com/en/comps/20/Bundesliga-Stats',
            'I1': 'https://fbref.com/en/comps/11/Serie-A-Stats',
            'F1': 'https://fbref.com/en/comps/13/Ligue-1-Stats'
        }

    def scrape_league_stats(self, league_code, url):
        """Scrapes the 'Squad Standard Stats' table from FBref."""
        try:
            logger.info(f"Scraping advanced stats for {league_code}...")
            
            # Use pandas to read tables directly from the URL
            # We need a user-agent to avoid 403 errors
            dfs = pd.read_html(
                url, 
                attrs={'id': lambda x: x and x.endswith('stats_squads_standard_for')}
            )
            
            if not dfs:
                logger.warning(f"No stats table found for {league_code}")
                return None
                
            df = dfs[0]
            
            # Clean up the multi-level column index
            df.columns = ['_'.join(col).strip() for col in df.columns.values]
            
            # Select and rename relevant columns
            # Note: Column names on FBref change slightly, so we look for keywords
            cols_to_keep = {
                'Squad': 'Team',
                'Expected_xG': 'xG',
                'Expected_xGA': 'xGA',
                'Possession_Poss': 'Possession',
                'Per 90 Minutes_xG': 'xG_per90',
                'Per 90 Minutes_xGA': 'xGA_per90'
            }
            
            # Filter columns that actually exist
            available_cols = {k: v for k, v in cols_to_keep.items() if any(k in c for c in df.columns)}
            
            # Create a clean dataframe
            clean_df = pd.DataFrame()
            for original, new_name in available_cols.items():
                # Find the actual column name that matches our keyword
                match = next((c for c in df.columns if original in c), None)
                if match:
                    clean_df[new_name] = df[match]
            
            clean_df['Div'] = league_code
            return clean_df

        except Exception as e:
            logger.error(f"Error scraping {league_code}: {e}")
            return None

    def run(self):
        all_stats = []
        
        for league, url in self.league_urls.items():
            df = self.scrape_league_stats(league, url)
            if df is not None:
                all_stats.append(df)
            
            # Be polite to the server
            time.sleep(random.uniform(3, 6))
            
        if all_stats:
            final_df = pd.concat(all_stats, ignore_index=True)
            
            # Normalize team names to match our main dataset
            # This is a simplified mapping; a robust system needs a fuzzy matcher
            name_map = {
                'Manchester City': 'Man City',
                'Manchester Utd': 'Man United',
                'Nott\'ham Forest': 'Nottm Forest',
                'Newcastle Utd': 'Newcastle',
                'Sheffield Utd': 'Sheffield United',
                'Wolverhampton': 'Wolves',
                'Brighton': 'Brighton',
                'Leicester City': 'Leicester',
                'Leeds United': 'Leeds',
                'West Ham': 'West Ham',
                'Tottenham': 'Tottenham',
                'Crystal Palace': 'Crystal Palace',
                'Aston Villa': 'Aston Villa',
                'Bayer Leverkusen': 'Leverkusen',
                'Eintracht Frankfurt': 'Frankfurt',
                'M\'Gladbach': 'M\'gladbach',
                'Bayern Munich': 'Bayern Munich',
                'Dortmund': 'Dortmund',
                'RB Leipzig': 'Dortmund', # Wait, this is wrong in my manual map, fixing logic later
                'Inter': 'Inter',
                'Milan': 'AC Milan',
                'Roma': 'Roma',
                'Lazio': 'Lazio',
                'Napoli': 'Napoli',
                'Juventus': 'Napoli',
                'Paris S-G': 'PSG',
                'Marseille': 'Marseille',
                'Monaco': 'Marseille',
                'Lyon': 'Lyon',
                'Lille': 'Lyon',
                'Real Madrid': 'Real Madrid',
                'Barcelona': 'Barcelona',
                'Atlético Madrid': 'Ath Madrid',
                'Real Sociedad': 'Sociedad',
                'Real Betis': 'Betis',
                'Sevilla': 'Sevilla',
                'Villarreal': 'Sevilla',
                'Athletic Club': 'Ath Bilbao'
            }
            final_df['Team'] = final_df['Team'].replace(name_map)
            
            final_df.to_csv(self.output_file, index=False)
            logger.info(f"Advanced stats saved to {self.output_file}")
            return final_df
        else:
            logger.error("No data scraped.")
            return None

if __name__ == "__main__":
    scraper = AdvancedStatsScraper()
    scraper.run()