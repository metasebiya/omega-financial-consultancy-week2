import cx_Oracle

# Sample connection string
dsn = cx_Oracle.makedsn("localhost", 1521, service_name="XEPDB1")
connection = cx_Oracle.connect(user="SYSTEM", password="July27,2022-GC", dsn=dsn)
cursor = connection.cursor()

# Function to get or insert bank and return bank_id
def get_or_create_bank_id(bank_name):
    cursor.execute("SELECT bank_id FROM banks WHERE bank_name = :name", {'name': bank_name})
    result = cursor.fetchone()
    if result:
        return result[0]
    else:
        cursor.execute("INSERT INTO banks (bank_name) VALUES (:name) RETURNING bank_id INTO :id",
                       {'name': bank_name, 'id': cursor.var(cx_Oracle.NUMBER)})
        bank_id = cursor.getimplicitresults()[0][0]
        return int(bank_id)

# Insert cleaned_df into database
def insert_reviews(cleaned_df):
    for _, row in cleaned_df.iterrows():
        bank_id = get_or_create_bank_id(row['bank_name'])
        cursor.execute("""
            INSERT INTO reviews (
                bank_id, review_text, sentiment, score, rating,
                identified_themes, spacy_keywords, tfidf_keywords
            ) VALUES (
                :bank_id, :review_text, :sentiment, :score, :rating,
                :identified_themes, :spacy_keywords, :tfidf_keywords
            )
        """, {
            'bank_id': bank_id,
            'review_text': row['review_text'],
            'sentiment': row['sentiment'],
            'score': row['score'],
            'rating': row['rating'],
            'identified_themes': ', '.join(row['identified_themes']) if isinstance(row['identified_themes'], list) else row['identified_themes'],
            'spacy_keywords': ', '.join(row['spacy_keywords']) if isinstance(row['spacy_keywords'], list) else row['spacy_keywords'],
            'tfidf_keywords': ', '.join(row['tfidf_keywords']) if isinstance(row['tfidf_keywords'], list) else row['tfidf_keywords']
        })

    connection.commit()

# Usage: call this inside your main loop
# insert_reviews(cleaned_df)
