import os
from pathlib import Path

# Define the file contents
files = {
    "requirements.txt": """pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
xgboost>=2.0.0
fastapi>=0.100.0
uvicorn>=0.23.0
requests>=2.31.0
python-dotenv>=1.0.0
""",
    ".env": """API_FOOTBALL_KEY=your_api_key_here
API_FOOTBALL_HOST=v3.football.api-sports.io
FOOTBALL_API_URL=https://v3.football.api-sports.io
""",
    ".gitignore": """# Python
__pycache__/
*.py[cod]

# Environment Variables
.env

# Data & Models
data/
models/
""",
    "README.md": """# Smart Football AI Match Predictor

An AI-powered tool that predicts football match outcomes using XGBoost, historical form, and weather data.

## Setup

1. **Install Dependencies**
