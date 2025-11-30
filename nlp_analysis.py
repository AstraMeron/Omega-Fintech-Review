"""
NLP Analysis Script
Task 2: Sentiment and Thematic Analysis

This script loads the processed reviews, applies:
1. Sentiment Analysis using DistilBERT (positive/negative/neutral).
2. Thematic Analysis using TF-IDF for keyword extraction.
"""
import pandas as pd
import numpy as np
import os
import re
import logging
from config import DATA_PATHS, BANK_NAMES
from tqdm import tqdm

# NLP Libraries
from transformers import pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# --- 1. CONFIGURATION AND SETUP ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# NLTK Data check (Ensure you run nltk.download('punkt'), nltk.download('stopwords'), nltk.download('wordnet'))
STOP_WORDS = set(stopwords.words('english'))
LEMMATIZER = WordNetLemmatizer()


class ReviewAnalyzer:
    """Class for performing NLP-based analysis on bank reviews."""

    def __init__(self):
        """Initializes analyzer with data paths and NLP pipeline."""
        self.input_path = DATA_PATHS['processed_reviews']
        self.output_path = DATA_PATHS['sentiment_results']
        self.df = pd.DataFrame()
        
        # Initialize DistilBERT Sentiment Pipeline
        logging.info("Initializing DistilBERT Sentiment Model...")
        self.sentiment_pipe = pipeline(
            "sentiment-analysis", 
            model="distilbert-base-uncased-finetuned-sst-2-english", 
            device=-1  # Use -1 for CPU, 0 for first GPU if available
        )
        logging.info("Model initialized successfully.")

    def load_data(self):
        """Load the processed review data."""
        try:
            self.df = pd.read_csv(self.input_path)
            # Ensure text is not null and is string
            self.df.dropna(subset=['review_text'], inplace=True)
            self.df['review_text'] = self.df['review_text'].astype(str)
            
            # Map bank_code to full name for clearer output
            self.df['bank_name'] = self.df['bank_code'].map(BANK_NAMES)
            
            logging.info(f"Loaded data successfully. Total reviews for analysis: {len(self.df)}")
            return True
        except FileNotFoundError:
            logging.error(f"Error: Input file not found at {self.input_path}. Run preprocessing first.")
            return False
        except KeyError as e:
            logging.error(f"Missing required column or config entry: {e}. Check 'bank_code' in reviews_processed.csv and BANK_NAMES in config.py.")
            return False

    # --- 2. SENTIMENT ANALYSIS ---

    def analyze_sentiment(self):
        """Performs sentiment analysis using DistilBERT."""
        logging.info("Starting Sentiment Analysis using DistilBERT...")

        texts = self.df['review_text'].tolist()
        results = self.sentiment_pipe(texts, batch_size=32)

        final_sentiment = []
        raw_model_scores = []
        
        for index, row in self.df.iterrows():
            res = results[index] 
            raw_label = res['label'].lower()
            raw_score = res['score']
            rating = row['rating']

            raw_model_scores.append(raw_score)

            # Rule-based adaptation for NEUTRAL
            if rating == 3 and raw_score < 0.75:
                final_sentiment.append('neutral')
            elif raw_label == 'positive':
                final_sentiment.append('positive')
            else:
                final_sentiment.append('negative')

        self.df['sentiment_label'] = final_sentiment
        self.df['sentiment_raw_score'] = raw_model_scores
        self.df['sentiment_score'] = self.df['rating']  # Use rating as main score

        logging.info("Sentiment Analysis complete (including rule-based Neutral handling).")

    # --- 3. THEMATIC ANALYSIS HELPERS ---

    def _preprocess_text_for_tfidf(self, text):
        """Tokenize, remove stop words, and lemmatize text."""
        text = text.lower()
        text = re.sub(r'[^a-z0-9\s]', '', text)

        try:
            tokens = word_tokenize(text)
        except LookupError:
            logging.error("NLTK 'punkt' resource not found. Please run nltk.download('punkt').")
            return ""

        processed_tokens = [
            LEMMATIZER.lemmatize(token)
            for token in tokens if token not in STOP_WORDS and len(token) > 2
        ]
        return " ".join(processed_tokens)


    def analyze_themes(self):
        """Performs Thematic Analysis using TF-IDF for keyword extraction."""
        logging.info("Starting Thematic Analysis via TF-IDF Keyword Extraction...")

        negative_neutral_df = self.df[self.df['sentiment_label'].isin(['negative', 'neutral'])].copy().reset_index(drop=True)
        
        if negative_neutral_df.empty:
            logging.warning("No Negative or Neutral reviews found to analyze themes.")
            self.df['top_keywords'] = 'N/A'
            return

        tqdm.pandas(desc="Pre-processing for TF-IDF")
        negative_neutral_df['processed_text'] = negative_neutral_df['review_text'].progress_apply(self._preprocess_text_for_tfidf)

        vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=5000)
        tfidf_matrix = vectorizer.fit_transform(negative_neutral_df['processed_text'])
        feature_names = vectorizer.get_feature_names_out()

        def extract_top_n_keywords(row_index, n=3):
            feature_array = tfidf_matrix[row_index].toarray().flatten()
            top_indices = np.argsort(feature_array)[::-1][:n]
            top_keywords = [feature_names[i] for i in top_indices if feature_array[i] > 0]
            return ", ".join(top_keywords)

        negative_neutral_df['top_keywords'] = [
            extract_top_n_keywords(i) for i in tqdm(range(len(negative_neutral_df)), desc="Extracting keywords")
        ]

        # Merge back safely
        self.df = pd.merge(
            self.df,
            negative_neutral_df[['review_text', 'top_keywords']],
            on='review_text',
            how='left'
        )
        self.df['top_keywords'] = self.df['top_keywords'].fillna('N/A (Positive Review)')

        logging.info("TF-IDF Keyword Extraction complete.")
        self.generate_thematic_report(negative_neutral_df, vectorizer, feature_names, tfidf_matrix)

    def generate_thematic_report(self, nn_df, vectorizer, feature_names, tfidf_matrix, k=10):
        """Generates a report of top keywords per bank."""
        print("\n" + "=" * 60)
        print("THEMATIC KEYWORD REPORT (FOR MANUAL CLUSTERING)")
        print("=" * 60)
        print("Use these keywords to manually group reviews into the 3-5 required themes.\n")

        for bank_code, bank_name in BANK_NAMES.items():
            bank_mask = nn_df['bank_code'] == bank_code
            if not bank_mask.any():
                print(f"--- {bank_name} --- \nNo negative/neutral reviews for keyword extraction.\n")
                continue

            bank_iloc_indices = bank_mask[bank_mask].index.to_numpy()
            bank_tfidf_sum = tfidf_matrix[bank_iloc_indices, :].sum(axis=0)
            bank_tfidf_scores = bank_tfidf_sum.A[0]
            sorted_indices = np.argsort(bank_tfidf_scores)[::-1]
            top_keywords = [feature_names[i] for i in sorted_indices[:k] if bank_tfidf_scores[i] > 0]

            print(f"--- {bank_name} ({len(bank_iloc_indices)} Negative/Neutral Reviews) ---")
            print(f"Top {k} Keywords (TF-IDF): {', '.join(top_keywords)}\n")

        print("=" * 60)


    def save_results(self):
        """Save the DataFrame with sentiment and thematic results."""
        final_cols = [
            'bank_code', 'bank_name', 'review_date', 'review_text', 'rating',
            'sentiment_label', 'sentiment_score', 'top_keywords'
        ]
        
        # Keep only existing final columns
        self.df = self.df[[col for col in final_cols if col in self.df.columns]]
        
        try:
            os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
            self.df.to_csv(self.output_path, index=False)
            logging.info(f"Results saved successfully to {self.output_path}. Total rows: {len(self.df)}.")
            return True
        except Exception as e:
            logging.error(f"Error saving data: {e}")
            return False

    def process(self):
        """Run the full analysis pipeline."""
        print("\n" + "=" * 60)
        print("STARTING TASK 2: NLP ANALYSIS")
        print("=" * 60)

        if not self.load_data():
            return False
        
        self.analyze_sentiment()
        self.analyze_themes()
        
        if self.save_results():
            print("\n✓ Task 2 NLP Analysis completed successfully!")
            print("------------------------------------------------------------")
            print(f"ACTION REQUIRED: Review the 'THEMATIC KEYWORD REPORT' above and the 'top_keywords' column in {self.output_path} for theme clustering.")
            return True
        return False

def main():
    analyzer = ReviewAnalyzer()
    analyzer.process()

if __name__ == "__main__":
    main()
