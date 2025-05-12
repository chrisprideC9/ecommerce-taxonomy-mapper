import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Application settings
DEFAULT_SIMILARITY_THRESHOLD = 0.3
DEFAULT_TEXT_WEIGHT = 0.7
DEFAULT_BATCH_SIZE = 100