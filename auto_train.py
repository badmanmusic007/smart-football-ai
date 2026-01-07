import logging
import time
from datetime import datetime
from pathlib import Path
import os

# Import the training function
from train_model import train_real_model
from src.config import settings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("training.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("AutoTrainer")

def run_auto_training():
    """
    Orchestrates the automated self-learning process.
    """
    logger.info("Initiating automated self-learning cycle...")
    
    try:
        # 1. Clear Cache to force fresh data download
        # We want the latest results from yesterday
        cache_file = settings.DATA_DIR / "match_data_final_v24.csv"
        if cache_file.exists():
            try:
                # Instead of deleting, we can just touch it to make it look old
                # or rely on the loader's logic. 
                # Better: Let's rename it to backup just in case
                backup_name = cache_file.with_suffix(f".bak_{int(time.time())}")
                os.rename(cache_file, backup_name)
                logger.info(f"Backed up old cache to {backup_name}")
            except Exception as e:
                logger.warning(f"Could not backup cache: {e}")

        # 2. Run Training
        # This will download fresh data and retrain the models
        logger.info("Starting model retraining...")
        train_real_model()
        
        logger.info("✅ Self-learning cycle complete. Models updated.")
        return True

    except Exception as e:
        logger.error(f"❌ Auto-training failed: {e}")
        return False

if __name__ == "__main__":
    run_auto_training()