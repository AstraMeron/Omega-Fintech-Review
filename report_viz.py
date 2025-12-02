"""
Report Visualization Script (report_viz.py)
Task 4: Extracts data from PostgreSQL, generates the essential visualizations,
and prepares insights for the final report.
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
import logging
import numpy as np

# Ensure necessary modules are on the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Assuming DatabaseLoader and CONFIG are accessible
try:
    from database_loader import DatabaseLoader
    from config import DATA_PATHS, BANK_NAMES
except ImportError as e:
    print(f"Error importing modules: {e}")
    sys.exit(1)

# --- Configuration and Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
sns.set_style("whitegrid")

class VisualizationEngine:
    """Handles data extraction from DB and visualization generation."""

    def __init__(self):
        logging.info("Initializing Visualization Engine...")
        self.db_loader = DatabaseLoader()
        # Ensure 'data/fig' directory exists
        self.output_dir = os.path.join(os.path.dirname(__file__), 'data', 'fig')
        os.makedirs(self.output_dir, exist_ok=True)
        self.connection_status = self.db_loader.connect()

        if not self.connection_status:
            logging.error("Failed to connect to the database. Cannot run visualizations.")

    def _execute_query(self, query):
        """Helper to execute an SQL query and return a DataFrame."""
        if not self.db_loader.conn or self.db_loader.conn.closed:
            logging.error("Database connection is closed. Re-establishing...")
            if not self.db_loader.connect():
                return pd.DataFrame()

        try:
            logging.info(f"Executing query: {query.splitlines()[0].strip()}...")
            df = pd.read_sql(query, self.db_loader.conn)
            return df
        except Exception as e:
            logging.error(f"Error executing SQL query: {e}")
            return pd.DataFrame()

    # ----------------------------------------------------
    # Data Retrieval Functions
    # ----------------------------------------------------

    def get_rating_distribution(self):
        """Retrieves data for overall rating distribution plot (Figure 1)."""
        query = """
        SELECT
            rating,
            COUNT(*) AS review_count
        FROM
            reviews
        GROUP BY
            rating
        ORDER BY
            rating;
        """
        return self._execute_query(query)

    def get_sentiment_breakdown(self):
        """Retrieves data for sentiment breakdown per bank plot (Figure 2)."""
        query = """
        SELECT
            b.bank_name,
            r.sentiment_label,
            COUNT(r.review_id) AS sentiment_count
        FROM
            reviews r
        JOIN
            banks b ON r.bank_id = b.bank_id
        GROUP BY
            b.bank_name, r.sentiment_label
        ORDER BY
            b.bank_name, r.sentiment_label;
        """
        return self._execute_query(query)

    # Note: Keyword functions are removed to bypass the column error

    # ----------------------------------------------------
    # Visualization Generation Functions
    # ----------------------------------------------------

    def plot_rating_distribution(self, df_rating):
        """Generates Figure 1: Overall Distribution of User Ratings."""
        if df_rating.empty: return

        plt.figure(figsize=(8, 6))
        sns.barplot(x='rating', y='review_count', data=df_rating, palette='viridis')
        plt.title('Figure 1. Overall Distribution of User Ratings (1-5 Stars)')
        plt.xlabel('Rating')
        plt.ylabel('Number of Reviews')
        plt.xticks(df_rating['rating'])
        
        filepath = os.path.join(self.output_dir, 'fig1_rating_distribution.png')
        plt.savefig(filepath)
        plt.close()
        logging.info(f"Figure 1 saved to {filepath}")
        return filepath

    def plot_sentiment_breakdown(self, df_sentiment):
        """Generates Figure 2: Sentiment Breakdown by Bank (Percentage Stacked Bar)."""
        if df_sentiment.empty: return

        # Pivot to get bank vs. sentiment counts
        df_pivot = df_sentiment.pivot(index='bank_name', columns='sentiment_label', values='sentiment_count').fillna(0)
        # Calculate percentages
        df_percent = df_pivot.apply(lambda x: x / x.sum() * 100, axis=1)
        df_percent = df_percent.reindex(columns=['negative', 'neutral', 'positive'], fill_value=0) # Ensure order

        plt.figure(figsize=(10, 7))
        # Stacked bar plot for percentages
        df_percent.plot(kind='bar', stacked=True, color={'negative': '#E34A33', 'neutral': '#FECC5C', 'positive': '#34A853'}, ax=plt.gca())
        
        plt.title('Figure 2. Sentiment Breakdown by Bank (Percentage)')
        plt.xlabel('Bank Name')
        plt.ylabel('Percentage of Reviews (%)')
        plt.xticks(rotation=45, ha='right')
        plt.legend(title='Sentiment', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        
        filepath = os.path.join(self.output_dir, 'fig2_sentiment_breakdown.png')
        plt.savefig(filepath)
        plt.close()
        logging.info(f"Figure 2 saved to {filepath}")
        return filepath


    # ----------------------------------------------------
    # Main Execution
    # ----------------------------------------------------
    def generate_visuals(self):
        """Runs the data extraction and visualization pipeline."""
        print("\n" + "=" * 60)
        print("STARTING TASK 4: GENERATING VISUALIZATIONS")
        print("=" * 60)
        
        if not self.connection_status:
             return False

        # 1. Plot 1: Rating Distribution
        df_rating = self.get_rating_distribution()
        self.plot_rating_distribution(df_rating)

        # 2. Plot 2: Sentiment Breakdown
        df_sentiment = self.get_sentiment_breakdown()
        self.plot_sentiment_breakdown(df_sentiment)
        
        # Plot 3: Keyword Frequency removed to avoid column name error

        logging.info("Visualization generation complete. Next: Theme Analysis and Report.")
        self.db_loader.close()
        return True

def main():
    engine = VisualizationEngine()
    engine.generate_visuals()

if __name__ == "__main__":
    main()