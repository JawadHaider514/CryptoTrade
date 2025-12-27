import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Feature Flags
USE_PRO_ANALYZER = os.getenv("USE_PRO_ANALYZER", "true").lower() == "true"
ENABLE_ORDERFLOW = os.getenv("ENABLE_ORDERFLOW", "false").lower() == "true"

# Signal Configuration
SIGNAL_REFRESH_INTERVAL = int(os.getenv("SIGNAL_REFRESH_INTERVAL", "30"))
SIGNAL_VALID_MINUTES = int(os.getenv("SIGNAL_VALID_MINUTES", "240"))

# Minimum Thresholds (temporarily lowered for UI testing with fallback signals)
MIN_CONFIDENCE = int(os.getenv("MIN_CONFIDENCE", "15"))
MIN_ACCURACY = int(os.getenv("MIN_ACCURACY", "0"))  # Set to 0 to show fallback signals with low accuracy

# Other Configuration
FLASK_PORT = int(os.getenv("FLASK_PORT", "5000"))
FLASK_HOST = os.getenv("FLASK_HOST", "0.0.0.0")
DEBUG_MODE = os.getenv("DEBUG_MODE", "false").lower() == "true"

# Flask App Configuration Dictionary (for backward compatibility with main.py)
APP_CONFIG = {
    "HOST": FLASK_HOST,
    "PORT": FLASK_PORT,
    "DEBUG": DEBUG_MODE,
    "USE_PRO_ANALYZER": USE_PRO_ANALYZER,
    "ENABLE_ORDERFLOW": ENABLE_ORDERFLOW,
    "SIGNAL_REFRESH_INTERVAL": SIGNAL_REFRESH_INTERVAL,
    "SIGNAL_VALID_MINUTES": SIGNAL_VALID_MINUTES,
    "MIN_CONFIDENCE": MIN_CONFIDENCE,
    "MIN_ACCURACY": MIN_ACCURACY,
}
