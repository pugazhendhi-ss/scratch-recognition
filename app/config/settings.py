import os
from pathlib import Path
from inference import get_model

from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Directory setup
BASE_DIR = Path(__file__).resolve().parent.parent.parent

OUTPUT_DIR = BASE_DIR / "app" / "utils" / "images"
OUTPUT_DIR.mkdir(exist_ok=True)

UPLOADED_VIDEO_DIR = BASE_DIR / "app" / "utils" / "video"
UPLOADED_VIDEO_DIR.mkdir(exist_ok=True)


# API Keys
ROBOFLOW_API_KEY = os.getenv("ROBOFLOW_API_KEY")
if not ROBOFLOW_API_KEY:
    raise ValueError("ROBOFLOW_API_KEY environment variable is required")

# Load the model only once
LOADED_MODEL = get_model(
               model_id="scratch-dent-lypwi/3",
               api_key=ROBOFLOW_API_KEY
               )



