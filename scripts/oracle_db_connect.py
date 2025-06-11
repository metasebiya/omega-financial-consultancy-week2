import os
import sys
import cx_Oracle
import pandas as pd
import glob
from dotenv import load_dotenv
# Allow import from src
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from load_data import DataLoader

# Load environment variables from .env file
load_dotenv()

# Fetch credentials from env variables
ORACLE_USER = os.getenv("ORACLE_USER")
ORACLE_PASSWORD = os.getenv("ORACLE_PASSWORD")
ORACLE_HOST = os.getenv("ORACLE_HOST", "localhost")
ORACLE_PORT = os.getenv("ORACLE_PORT", "1521")
ORACLE_SERVICE = os.getenv("ORACLE_SERVICE", "XEPDB1")

# Create DSN
dsn = cx_Oracle.makedsn(ORACLE_HOST, ORACLE_PORT, service_name=ORACLE_SERVICE)

# Connect
conn = cx_Oracle.connect(user=ORACLE_USER, password=ORACLE_PASSWORD, dsn=dsn)
cursor = conn.cursor()

cursor.execute("SELECT USER FROM dual")
current_user = cursor.fetchone()[0]
print(f"Connected as user: {current_user}")


# Your insert logic
bank_id_map = {}
csv_files_map = {
    "Commercial Bank of Ethiopia": "../data/processed/CBE_reviews_processed_for_themes.csv",
    "Bank of Abysinnia": "../data/processed/BOA_reviews_processed_for_themes.csv",
    "Dashen Bank": "../data/processed/Dashen_reviews_processed_for_themes.csv"
}

loader = DataLoader()
data_sources = loader.load_data(csv_files_map)
for key, df in data_sources.items():

    bank_name = df['bank_name'].iloc[0] if 'bank_name' in df.columns else key

    if bank_name not in bank_id_map:
        # Create the bind variable first
        out_id = cursor.var(cx_Oracle.NUMBER)

        # Execute with the bind variables (input + output)
        cursor.execute(
            "INSERT INTO BANK (bank_name) VALUES (:1) RETURNING bank_id INTO :2",
            [bank_name, out_id]
        )
        bank_id = cursor.fetchone()[0]
        bank_id_map[bank_name] = bank_id
    else:
        bank_id = bank_id_map[bank_name]

    for _, row in df.iterrows():
        cursor.execute("""
            INSERT INTO REVIEW (
                bank_id, review_text, sentiment, review_date, score,
                identified_themes, rating, spacy_keywords, tfidf_keywords
            ) VALUES (
                :1, :2, :3, TO_DATE(:4, 'YYYY-MM-DD'), :5, :6, :7, :8, :9
            )
        """, (
            bank_id,
            row['review_text'],
            row['sentiment'],
            str(row['date']),
            row['score'],
            row['identified_themes'],
            row['rating'],
            row['spacy_keywords'],
            row['tfidf_keywords']
        ))

# Commit & close
conn.commit()
cursor.close()
conn.close()
print("✅ Data load complete.")
