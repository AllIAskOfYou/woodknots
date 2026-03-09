import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Define standard variables with fallbacks
RAW_DIR = Path(os.getenv("RAW_DIR", "./data/raw"))
PROCESSED_DIR = Path(os.getenv("PROCESSED_DIR", "./data/processed"))
METADATA_DIR = Path(os.getenv("METADATA_DIR", "./metadata"))
EXPERIMENTS_DIR = Path(os.getenv("EXPERIMENTS_DIR", "./experiments"))
