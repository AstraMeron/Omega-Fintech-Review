"""
Configuration file for Bank Reviews Analysis Project
"""
import os
from dotenv import load_dotenv

# Load environment variables (ensure you have a .env file or rely on defaults)
load_dotenv()

# Google Play Store App IDs
# Note: Using the primary mobile banking apps for BOA and Dashen.
APP_IDS = {
    # Commercial Bank of Ethiopia (CBE)
    'CBE': os.getenv('CBE_APP_ID', 'com.combanketh.mobilebanking'),
    # Bank of Abyssinia (BOA) - using the 'BoA Mobile' app
    'BOA': os.getenv('BOA_APP_ID', 'com.boa.boaMobileBanking'),
    # Dashen Bank (DASHEN) - using the 'Dashen Bank Super App'
    'DASHEN': os.getenv('DASHEN_APP_ID', 'com.dashen.dashensuperapp')
}

# Bank Names Mapping
BANK_NAMES = {
    'CBE': 'Commercial Bank of Ethiopia',
    'BOA': 'Bank of Abyssinia',
    'DASHEN': 'Dashen Bank'
}

# Scraping Configuration
SCRAPING_CONFIG = {
    # Target 400 reviews per bank, as specified in the challenge
    'reviews_per_bank': int(os.getenv('REVIEWS_PER_BANK', 400)),
    'max_retries': int(os.getenv('MAX_RETRIES', 3)),
    'lang': 'en',  # Language filter (English)
    'country': 'et' # Country filter (Ethiopia)
}

# File Paths
DATA_PATHS = {
    'raw': 'data/raw',
    'processed': 'data/processed',
    'raw_reviews': 'data/raw/reviews_raw.csv',
    'processed_reviews': 'data/processed/reviews_processed.csv',
    'sentiment_results': 'data/processed/reviews_with_sentiment.csv',
    'final_results': 'data/processed/reviews_final.csv'
}