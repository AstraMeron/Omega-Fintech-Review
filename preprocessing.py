"""
Data Preprocessing Script
Task 1: Data Preprocessing

This script cleans and preprocesses the scraped reviews data.
- Handles missing values
- Normalizes dates
- Cleans text data
- NEW: Removes duplicates
- NEW: Removes Amharic/non-ASCII characters
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime
import re
from config import DATA_PATHS
import logging

# Configure basic logging for better feedback
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class ReviewPreprocessor:
    """Preprocessor class for review data"""

    def __init__(self, input_path=None, output_path=None):
        """Initialization"""
        self.input_path = input_path if input_path else DATA_PATHS['raw_reviews']
        self.output_path = output_path if output_path else DATA_PATHS['processed_reviews']
        self.df = pd.DataFrame() # DataFrame to hold the data
        self.report = {} # Dictionary to store preprocessing metrics

    def load_data(self):
        """Load the raw reviews data from the specified input path."""
        try:
            self.df = pd.read_csv(self.input_path)
            logging.info(f"Data loaded successfully from {self.input_path}. Initial size: {len(self.df)} rows.")
            return True
        except FileNotFoundError:
            logging.error(f"Error: Input file not found at {self.input_path}. Run the scraper first.")
            return False
        except Exception as e:
            logging.error(f"Error loading data: {e}")
            return False

    def save_data(self):
        """Save the processed DataFrame to the specified output path."""
        try:
            # Ensure the directory exists
            os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
            self.df.to_csv(self.output_path, index=False)
            logging.info(f"Processed data saved successfully to {self.output_path}. Final size: {len(self.df)} rows.")
            return True
        except Exception as e:
            logging.error(f"Error saving data: {e}")
            return False

    def check_missing_data(self):
        """Analyze and report on missing values."""
        missing = self.df.isnull().sum()
        self.report['missing_values'] = missing.to_dict()
        logging.info("Missing data check complete.")

    def handle_missing_values(self):
        """Handle missing values by dropping rows where 'review_text' is NaN."""
        initial_len = len(self.df)
        
        # Drop rows where the critical column 'review_text' is missing
        self.df.dropna(subset=['review_text'], inplace=True)
        
        # Fill missing 'user_name' with a placeholder
        self.df['user_name'].fillna('Anonymous User', inplace=True)
        
        removed_count = initial_len - len(self.df)
        self.report['dropped_rows_missing_text'] = removed_count
        
        logging.info(f"Dropped {removed_count} rows with missing review text. Missing values handled.")

    def normalize_dates(self):
        """Convert 'review_date' to a consistent datetime format."""
        if 'review_date' in self.df.columns:
            # Attempt to convert to datetime objects
            self.df['review_date'] = pd.to_datetime(self.df['review_date'], errors='coerce')
            
            # Fill any NaT (Not a Time) values with the mode or a placeholder if necessary
            # For this challenge, we'll ensure we keep only valid dates or drop if date is critical
            # Assuming date validity is not critical for text analysis, we'll keep the rows.
            
            logging.info("Date normalization complete.")
        else:
            logging.warning("'review_date' column not found.")

    # ==========================================================
    # NEW REQUIRED FUNCTION: REMOVE DUPLICATES
    # ==========================================================
    def remove_duplicates(self):
        """Identify and remove duplicate reviews based on review_text and bank_code."""
        
        initial_len = len(self.df)
        
        # Drop duplicates based on the critical content columns: text and the bank it belongs to.
        self.df.drop_duplicates(subset=['review_text', 'bank_code'], inplace=True)
        
        removed_count = initial_len - len(self.df)
        self.report['removed_duplicates'] = removed_count
        
        logging.info(f"Removed {removed_count} duplicate reviews.")
    # ==========================================================


    # ==========================================================
    # NEW HELPER FUNCTION: REMOVE NON-ENGLISH (AMHARIC) CHARACTERS
    # ==========================================================
    def _remove_non_ascii(self, text):
        """
        Removes non-ASCII characters, which effectively removes most Amharic/Ge'ez script
        and other non-English characters.
        """
        # Ensure text is string and remove all characters outside the standard ASCII range (0-127)
        return re.sub(r'[^\x00-\x7F]+', '', str(text))
    # ==========================================================


    def clean_text(self):
        """Perform text cleaning steps on the 'review_text' column."""
        if 'review_text' in self.df.columns:
            
            # 1. Lowercasing
            self.df['review_text'] = self.df['review_text'].str.lower()
            
            # 2. NEW: Remove Amharic/non-ASCII characters
            self.df['review_text'] = self.df['review_text'].apply(self._remove_non_ascii)

            # 3. Remove extra whitespace
            self.df['review_text'] = self.df['review_text'].str.strip().str.replace(r'\s+', ' ', regex=True)
            
            # Remove rows where the text became empty after cleaning
            initial_len = len(self.df)
            self.df.replace('', np.nan, inplace=True)
            self.df.dropna(subset=['review_text'], inplace=True)
            dropped_empty = initial_len - len(self.df)
            self.report['dropped_rows_empty_text'] = dropped_empty

            logging.info("Text cleaning (including non-English character removal) complete.")
        else:
            logging.warning("'review_text' column not found.")
            
    def validate_ratings(self):
        """Ensure 'rating' is an integer type and within a valid range (1-5)."""
        if 'rating' in self.df.columns:
            # Coerce the rating to numeric, setting errors to NaN
            self.df['rating'] = pd.to_numeric(self.df['rating'], errors='coerce').astype('Int64')
            
            # Drop rows where rating is invalid (outside 1-5 range)
            initial_len = len(self.df)
            self.df.dropna(subset=['rating'], inplace=True)
            self.df = self.df[(self.df['rating'] >= 1) & (self.df['rating'] <= 5)]
            
            removed_count = initial_len - len(self.df)
            self.report['removed_invalid_ratings'] = removed_count
            
            logging.info(f"Removed {removed_count} rows with invalid ratings. Ratings validated.")
        else:
            logging.warning("'rating' column not found.")

    def prepare_final_output(self):
        """Select and reorder final columns for the processed DataFrame."""
        
        # List of columns to keep in the final processed data
        final_cols = [
            'bank_code', 
            'review_date', 
            'review_text', 
            'rating', 
            'user_name', 
            'thumbs_up_count'
        ]
        
        # Keep only the final required columns
        self.df = self.df.reindex(columns=final_cols)
        logging.info("Final output columns prepared.")

    def generate_report(self):
        """Print a summary report of the preprocessing steps."""
        print("\n" + "=" * 60)
        print("PREPROCESSING SUMMARY REPORT")
        print("=" * 60)
        
        print(f"Initial Review Count: {self.report.get('initial_len', 'N/A')}")
        print(f"Final Review Count: {len(self.df)}")
        
        print("\n--- Rows Removed ---")
        print(f"Duplicates Removed: {self.report.get('removed_duplicates', 0)}")
        print(f"Rows Dropped (Missing Text): {self.report.get('dropped_rows_missing_text', 0)}")
        print(f"Rows Dropped (Empty Text after Cleaning): {self.report.get('dropped_rows_empty_text', 0)}")
        print(f"Rows Dropped (Invalid Ratings): {self.report.get('removed_invalid_ratings', 0)}")
        
        print("\n--- Data Quality ---")
        print(f"Date Column Type: {self.df['review_date'].dtype if 'review_date' in self.df.columns else 'N/A'}")
        print(f"Rating Column Type: {self.df['rating'].dtype if 'rating' in self.df.columns else 'N/A'}")
        print("Text cleaning includes lowercasing, non-ASCII removal, and whitespace normalization.")
        print("=" * 60)
        
    def process(self):
        """Run the full preprocessing pipeline."""
        self.report['initial_len'] = len(self.df) # Initialize before loading
        print("=" * 60)
        print("STARTING DATA PREPROCESSING")
        print("=" * 60)

        if not self.load_data():
            return False
        
        self.report['initial_len'] = len(self.df)

        # Run each step of the pipeline in sequence
        self.check_missing_data()
        
        # --- NEW REQUIRED STEP (Re-added): REMOVE DUPLICATES ---
        self.remove_duplicates() 
        # ------------------------------------------------------

        self.handle_missing_values()
        self.normalize_dates()
        
        # --- CLEAN TEXT (Now includes Amharic removal) ---
        self.clean_text() 
        # -------------------------------------------------
        
        self.validate_ratings()
        self.prepare_final_output()

        if self.save_data():
            self.generate_report()
            return True

        return False


def main():
    """Main execution function"""
    preprocessor = ReviewPreprocessor()
    success = preprocessor.process()

    if success:
        print("\n✓ Preprocessing completed successfully!")
        return preprocessor.df
    else:
        print("\n✗ Preprocessing failed!")
        return None

if __name__ == "__main__":
    
    # Check if the script is being run directly
    # To correctly handle module imports when running main.py from the root directory,
    # we need to ensure the project root is in the path.
    # The original file's path handling was complex, so we will simplify and assume
    # both config.py and preprocessing.py are in the root directory.
    
    try:
        # Assuming the main execution flow is from the root directory
        # where config.py is located.
        main()
    except Exception as e:
        print(f"An error occurred during main execution: {e}")
        # Optionally, check if config module failed to load and suggest a fix
        if 'config' in str(e):
             print("\nNote: If config.py failed to load, ensure it is in the same directory as preprocessing.py.")