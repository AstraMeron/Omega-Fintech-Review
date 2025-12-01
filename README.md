# 🏦 Customer Experience Analytics for Ethiopian Fintech Apps

## Project Overview

This project was developed for Omega Consultancy to analyze customer satisfaction and identify critical pain points for three major Ethiopian mobile banking applications:

* Commercial Bank of Ethiopia (CBE)
* Bank of Abyssinia (BOA)
* Dashen Bank

By collecting user reviews from the Google Play Store, the project aims to provide data-driven insights into system stability, performance, and user experience (UX) to guide future application development and product strategy.

## 🛠️ Setup and Installation

### Prerequisites

* Python 3.8+
* Git
* (For Task 3) PostgreSQL database instance

### Clone the Repository

```bash
git clone [https://github.com/AstraMeron/Omega-Fintech-Review]
cd Omega-Fintech-Review
```

### Environment Setup

It is highly recommended to use a virtual environment:

```bash
# Code block starts here:
python -m venv venv
```

# On Windows (use this command)
```bash
.\venv\Scripts\activate
```

# On macOS/Linux (use this command)
```bash
source venv/bin/activate
```


### Install Dependencies

```bash
pip install -r requirements.txt
```


The project follows a standard data engineering and analysis pipeline across four main tasks:

### Task Status Summary

* **Task 1: Data Collection & Preprocessing**
    * **Description:** Scraped reviews and cleaned raw text data.
    * **Techniques:** `google-play-scraper`, Pandas, Regex.
    * **Status:** Completed.

* **Task 2: NLP Analysis (Sentiment & Theme)**
    * **Description:** Quantified sentiment and extracted actionable themes (pain points).
    * **Techniques:** **DistilBERT**, **TF-IDF**, Manual Clustering.
    * **Status:** Completed.

* **Task 3: Database Storage**
    * **Description:** Persist the analyzed data in a relational database.
    * **Status:** Pending.

* **Task 4: Visualization & Reporting**
    * **Description:** Generate final visualizations and comprehensive report.
    * **Status:** Pending.

## 📊 Task 2: Key Findings (Interim Analysis)

The analysis was performed on 957 user reviews. Thematic analysis focused on the 436 Negative and Neutral reviews using DistilBERT for sentiment and TF-IDF for keyword clustering.

### Dominant Thematic Pain Points (3+ Themes Identified per Bank)

The analysis clustered user feedback into critical pain points, consistently finding that **App & System Stability** is the most urgent issue across the sector.

**1. App & System Stability (T1)**
* **Description:** Frequent crashes, errors, and the general inability to use the application reliably.
* **Affected Banks:** CBE, BOA, Dashen
* **Example Keywords:** `working`, `cant`, `worst app`, `doesnt`

**2. Performance & Speed (T2)**
* **Description:** Slow loading, lagging, delayed transactions, and excessive waiting times.
* **Affected Banks:** BOA, Dashen
* **Example Keywords:** `slow`, `time`

**3. Core Functionality & Access (T3)**
* **Description:** Problems completing core banking tasks such as transfers, viewing account history, or managing account access.
* **Affected Banks:** CBE
* **Example Keywords:** `transaction`, `account`, `history`

**4. User Experience (UX) & Quality (T4)**
* **Description:** General critique related to the app's look and feel, or dissatisfaction with recent feature updates.
* **Affected Banks:** CBE, BOA, Dashen
* **Example Keywords:** `app`, `update`, `bad`

### Database Schema (Task 3)

The project utilizes a PostgreSQL relational database (`bank_reviews`) with two tables to store and relate the analyzed data:

| Table Name | Column Name | Data Type | Key/Notes |
| :--- | :--- | :--- | :--- |
| **banks** | `bank_id` | SERIAL | PRIMARY KEY |
| | `bank_code` | VARCHAR(10) | UNIQUE (e.g., 'CBE') |
| | `bank_name` | VARCHAR(100) | |
| **reviews** | `review_id` | SERIAL | PRIMARY KEY |
| | `bank_id` | INTEGER | FOREIGN KEY (references banks) |
| | `review_text` | TEXT | |
| | `rating` | INTEGER | |
| | `review_date` | DATE | |
| | `sentiment_label` | VARCHAR(20) | |
| | `sentiment_score` | FLOAT | |
| | `source` | VARCHAR(50) | |