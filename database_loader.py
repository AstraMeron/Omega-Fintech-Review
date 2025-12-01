import pandas as pd
import psycopg2
from psycopg2 import sql
from dotenv import load_dotenv
import os
import sys

# Ensure config is accessible
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# Note: Ensure your config.py contains DATA_PATHS dict and BANK_NAMES dict
from config import DATA_PATHS, BANK_NAMES 

# Load environment variables (for DB credentials)
load_dotenv()

class DatabaseLoader:
    """Handles connecting to PostgreSQL and loading processed review data."""
    def __init__(self):
        # Database Credentials loaded from .env
        self.db_host = os.getenv('DB_HOST')
        self.db_port = os.getenv('DB_PORT')
        self.db_name = os.getenv('DB_NAME')
        self.db_user = os.getenv('DB_USER')
        self.db_password = os.getenv('DB_PASSWORD')
        self.conn = None
        self.cursor = None

        # Load the processed data
        # IMPORTANT: We assume the sentiment results file is in data/processed/
        self.df = pd.read_csv(DATA_PATHS['sentiment_results'])
        
        # Prepare the dataframe by creating a 'source' column (if missing)
        self.df['source'] = 'Google Play Store'

    def connect(self):
        """Establishes a connection to the PostgreSQL database."""
        try:
            self.conn = psycopg2.connect(
                host=self.db_host,
                port=self.db_port,
                database=self.db_name,
                user=self.db_user,
                password=self.db_password
            )
            self.conn.autocommit = True
            self.cursor = self.conn.cursor()
            print("✓ Successfully connected to PostgreSQL database.")
            return True
        except Exception as e:
            print(f"✗ Database connection failed: {e}")
            return False

    def create_tables(self):
        """Creates the 'banks' and 'reviews' tables with the specified schema."""
        
        # 1. Banks Table
        banks_table_sql = """
        CREATE TABLE IF NOT EXISTS banks (
            bank_id SERIAL PRIMARY KEY,
            bank_code VARCHAR(10) UNIQUE NOT NULL,
            bank_name VARCHAR(100) UNIQUE NOT NULL
        );
        """
        # 2. Reviews Table (FIXED: Removed source_review_id)
        reviews_table_sql = """
        CREATE TABLE IF NOT EXISTS reviews (
            review_id SERIAL PRIMARY KEY,
            bank_id INTEGER REFERENCES banks(bank_id),
            review_text TEXT,
            rating INTEGER NOT NULL,
            review_date DATE,
            sentiment_label VARCHAR(20),
            sentiment_score FLOAT,
            source VARCHAR(50)
        );
        """
        try:
            print("Creating tables...")
            self.cursor.execute(banks_table_sql)
            self.cursor.execute(reviews_table_sql)
            print("✓ Tables 'banks' and 'reviews' created or already exist.")
        except Exception as e:
            print(f"✗ Error creating tables: {e}")

    def insert_banks_data(self):
        """Inserts unique bank data into the 'banks' table."""
        unique_banks = self.df['bank_code'].unique()
        
        for code in unique_banks:
            name = BANK_NAMES.get(code, code)
            try:
                upsert_sql = """
                INSERT INTO banks (bank_code, bank_name)
                VALUES (%s, %s)
                ON CONFLICT (bank_code) DO NOTHING;
                """
                self.cursor.execute(upsert_sql, (code, name))
            except Exception as e:
                print(f"✗ Error inserting bank {name}: {e}")
                
        print(f"✓ Inserted/checked {len(unique_banks)} unique banks.")

    def insert_reviews_data(self):
        """Inserts processed review data into the 'reviews' table."""
        print("Starting review data insertion...")
        
        # 1. Retrieve the newly generated bank_ids for mapping
        self.cursor.execute("SELECT bank_id, bank_code FROM banks;")
        bank_id_map = {code: id for id, code in self.cursor.fetchall()}
        
        # 2. Add the bank_id FK column to the DataFrame
        self.df['bank_id_fk'] = self.df['bank_code'].map(bank_id_map)

        # 3. Prepare data for insertion (FIXED: Removed 'reviewId')
        cols_to_insert = [
            'bank_id_fk', 'review_text', 'rating', 'review_date', 
            'sentiment_label', 'sentiment_score', 'source'
        ]
        
        # Ensure review_date is in a format PostgreSQL accepts (YYYY-MM-DD)
        self.df['review_date'] = pd.to_datetime(self.df['review_date']).dt.strftime('%Y-%m-%d')
        
        data_tuples = [tuple(row) for row in self.df[cols_to_insert].values]
        
        # The INSERT statement must match the fixed column list
        insert_query = sql.SQL("""
            INSERT INTO reviews (bank_id, review_text, rating, review_date, 
                                 sentiment_label, sentiment_score, source)
            VALUES {}
        """).format(sql.SQL(',').join(sql.Literal(row) for row in data_tuples))

        try:
            # Execute the batch insert
            self.cursor.execute(insert_query)
            insert_count = self.cursor.rowcount 
            print(f"✓ Successfully inserted {insert_count} reviews.")
            return insert_count
        except Exception as e:
            print(f"✗ Error during batch review insertion: {e}")
            print(f"  Error details: {e}")
            return 0

    def run_verification_queries(self):
        """Runs simple queries to verify data integrity (KPI requirement)."""
        print("\n--- Running Verification Queries ---")
        
        # 1. Count reviews per bank
        query_count_per_bank = """
        SELECT b.bank_name, COUNT(r.review_id) 
        FROM reviews r
        JOIN banks b ON r.bank_id = b.bank_id
        GROUP BY 1 ORDER BY 2 DESC;
        """
        print("1. Review Count per Bank:")
        self.cursor.execute(query_count_per_bank)
        for row in self.cursor.fetchall():
            print(f"  {row[0]}: {row[1]} reviews")

        # 2. Calculate average rating per bank
        query_avg_rating = """
        SELECT b.bank_name, ROUND(AVG(r.rating), 2) AS avg_rating
        FROM reviews r
        JOIN banks b ON r.bank_id = b.bank_id
        GROUP BY 1 ORDER BY 2 DESC;
        """
        print("\n2. Average Star Rating per Bank:")
        self.cursor.execute(query_avg_rating)
        for row in self.cursor.fetchall():
            print(f"  {row[0]}: {row[1]}")
        
        print("--- Verification Complete ---")


    def close(self):
        """Closes the database connection."""
        if self.conn:
            self.conn.close()
            print("✓ Database connection closed.")

    def run_loader(self):
        """Main execution flow for the database loader."""
        if not self.connect():
            return

        # Always try to create tables first
        self.create_tables() 
        # Insert banks (must happen before inserting reviews)
        self.insert_banks_data()
        
        inserted_count = self.insert_reviews_data()
        
        if inserted_count >= 400: 
            print(f"\n✓ Task 3 Minimum Essential KPI ({inserted_count} >= 400 reviews) met.")
            self.run_verification_queries()
        else:
            print(f"\n✗ Insertion did not meet minimum review count KPI. Inserted {inserted_count} reviews.")

        self.close()

if __name__ == "__main__":
    loader = DatabaseLoader()
    loader.run_loader()