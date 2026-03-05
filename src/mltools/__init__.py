import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Define standard variables with fallbacks
DATA_DIR = Path(os.getenv("DATA_DIR", "./data"))
RAW_DATA = DATA_DIR / "raw"
PROCESSED_DATA = DATA_DIR / "processed"
