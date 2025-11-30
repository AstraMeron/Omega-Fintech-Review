import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import re
from config import DATA_PATHS, BANK_NAMES # Assuming config is accessible

# Set plotting style
sns.set_style("whitegrid")

def generate_visualizations():
    """Loads analyzed data and generates key visualizations for the Interim Report."""
    try:
        df = pd.read_csv(DATA_PATHS['sentiment_results'])
        print(f"Loaded {len(df)} reviews for visualization.")
    except FileNotFoundError:
        print(f"Error: Data file not found at {DATA_PATHS['sentiment_results']}. Please run nlp_analysis.py first.")
        return

    # --- Chart 1: Overall Rating Distribution ---
    plt.figure(figsize=(8, 5))
    rating_counts = df['rating'].value_counts().sort_index()
    sns.barplot(x=rating_counts.index, y=rating_counts.values, palette="viridis")
    plt.title('Figure 1. Overall Distribution of User Ratings (1-5 Stars)')
    plt.xlabel('Rating')
    plt.ylabel('Number of Reviews')
    plt.savefig(f"{DATA_PATHS['processed']}/fig1_rating_distribution.png")
    plt.close()
    print("Saved fig1_rating_distribution.png")

    # --- Chart 2: Sentiment Breakdown by Bank (Normalized) ---
    sentiment_df = df.groupby('bank_name')['sentiment_label'].value_counts(normalize=True).mul(100).unstack(fill_value=0)
    sentiment_df = sentiment_df[['negative', 'neutral', 'positive']] # Ensure consistent column order

    plt.figure(figsize=(10, 6))
    sentiment_df.plot(kind='bar', stacked=True, color={'negative': '#E74C3C', 'neutral': '#F9E79F', 'positive': '#2ECC71'}, ax=plt.gca())
    plt.title('Figure 2. Sentiment Breakdown by Bank (Percentage)')
    plt.ylabel('Percentage of Reviews (%)')
    plt.xlabel('Bank Name')
    plt.xticks(rotation=45, ha='right')
    plt.legend(title='Sentiment')
    plt.tight_layout()
    plt.savefig(f"{DATA_PATHS['processed']}/fig2_sentiment_breakdown.png")
    plt.close()
    print("Saved fig2_sentiment_breakdown.png")

    # --- Chart 3: Top 10 Overall Keywords (Aggregated from Negative/Neutral Reviews) ---
    negative_neutral_df = df[df['sentiment_label'].isin(['negative', 'neutral'])]
    
    # Safely combine all keywords from the 'top_keywords' column
    all_keywords = []
    for keywords_str in negative_neutral_df['top_keywords'].dropna():
        # Remove the 'N/A (Positive Review)' placeholder if any remain
        if "N/A" not in keywords_str:
            all_keywords.extend([kw.strip() for kw in keywords_str.split(',')])

    # Count the top 10 most frequent single words/n-grams
    word_counts = Counter(all_keywords)
    top_10 = pd.DataFrame(word_counts.most_common(10), columns=['Keyword', 'Frequency'])

    plt.figure(figsize=(10, 6))
    sns.barplot(x='Frequency', y='Keyword', data=top_10, palette="mako")
    plt.title('Figure 3. Top 10 Most Frequent Keywords (Across All Negative/Neutral Reviews)')
    plt.xlabel('Frequency')
    plt.ylabel('Keyword / N-gram')
    plt.tight_layout()
    plt.savefig(f"{DATA_PATHS['processed']}/fig3_top_keywords.png")
    plt.close()
    print("Saved fig3_top_keywords.png")

if __name__ == "__main__":
    generate_visualizations()