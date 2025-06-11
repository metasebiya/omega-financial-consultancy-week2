import pandas as pd
import emoji
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from load_data import DataLoader
from clean_data import DataCleaner
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer

class DataPreprocessor:
    bank_theme_mappings = {
        "CBE": {
            'Account Access Issues': ['login', 'otp', 'access', 'fingerprint', 'password', 'account'],
            'Transaction Performance': ['transfer', 'slow', 'fast', 'loading', 'payment', 'transaction'],
            'User Interface & Experience': ['ui', 'interface', 'design', 'bug', 'crash', 'smooth', 'easy to use'],
            'Customer Support': ['support', 'service', 'help', 'response'],
            'Feature Requests': ['feature', 'new', 'add', 'option', 'budgeting', 'bill pay']
        },
        "BOA": {
            'Account Access Issues': ['login', 'error', 'access', 'otp', 'fingerprint', 'account'],
            'Transaction Performance': ['transfer', 'slow', 'take long', 'transaction', 'payment'],
            'User Interface & Experience': ['ui', 'confusing', 'crash', 'unreliable', 'design', 'buggy'],
            'Customer Support': ['support', 'responsive', 'help', 'service'],
            'Feature Requests': ['feature', 'budgeting', 'new', 'add']
        },
        "Dashen": {
            'Account Access Issues': ['login', 'access', 'no issues', 'password'],
            'Transaction Performance': ['transfer', 'slow', 'smooth', 'balance', 'transaction'],
            'User Interface & Experience': ['ui', 'old-fashioned', 'functional', 'crash', 'optimization',
                                            'interface'],
            'Customer Support': ['support', 'excellent', 'helpful', 'team'],
            'Feature Requests': ['feature', 'bill pay', 'new', 'add']
        }
    }
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.analyzer = SentimentIntensityAnalyzer()
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            print("Downloading spaCy model 'en_core_web_sm'...")
            spacy.cli.download("en_core_web_sm")
            self.nlp = spacy.load("en_core_web_sm")

    @staticmethod
    def contains_emoji(text):
        return any(char in emoji.EMOJI_DATA for char in text)

    @staticmethod
    def get_sentiment_label(score):
        if score >= 0.05:
            return "positive"
        elif score <= -0.05:
            return "negative"
        else:
            return "neutral"

    def vader_sentiment_analysis(self, text):
        try:
            if not isinstance(text, str) or text.strip() == "":
                return pd.Series(["unknown", None, "none"])

            score = self.analyzer.polarity_scores(text)['compound']
            label = self.get_sentiment_label(score)
            return pd.Series([label, score, "vader"])

        except Exception as e:
            print(f"❌ Error processing text: {text[:40]}...\n{e}")
            return pd.Series(["error", None, "none"])

    def aggregate_sentiment_by_bank_rating(self):
        """
        Aggregates the mean sentiment score for each bank grouped by review rating.
        Assumes 'bank', 'rating', and 'score' columns exist after sentiment analysis.
        """
        if 'bank_name' not in self.df.columns or 'rating' not in self.df.columns or 'score' not in self.df.columns:
            print("Error: 'bank', 'rating', or 'score' column not found for aggregation.")
            return pd.DataFrame()

        # Ensure 'score' is numeric, coercing errors to NaN
        self.df['score'] = pd.to_numeric(self.df['score'], errors='coerce')

        # Group by bank and rating, then calculate the mean sentiment score
        mean_sentiment = self.df.groupby(['bank_name', 'rating'])['score'].mean().reset_index()
        mean_sentiment.rename(columns={'score': 'mean_sentiment_score'}, inplace=True)
        return mean_sentiment

    def extract_keywords_spacy(self, text):
        """
        Extracts keywords (noun chunks, relevant POS tags) and n-grams using spaCy.
        Lemmatization is applied for consistency.
        """
        if not isinstance(text, str) or not text.strip():
            return []

        doc = self.nlp(text.lower())
        keywords = []

        # Extract noun chunks (phrases that often represent key concepts)
        for chunk in doc.noun_chunks:
            # Filter out very short or generic noun chunks
            if len(chunk.text.split()) > 1 and chunk.text.strip() not in ['app', 'bank']:
                keywords.append(chunk.text.strip())
            elif len(chunk.text.split()) == 1 and len(chunk.text.strip()) > 2:  # Single word noun
                keywords.append(chunk.text.strip())

        # Extract individual tokens (lemmas) based on Part-of-Speech, excluding stop words and punctuation
        for token in doc:
            if token.pos_ in ['NOUN', 'ADJ', 'VERB'] and not token.is_stop and not token.is_punct:
                if len(token.lemma_) > 2 and token.lemma_ not in ['app', 'bank', 'get', 'use', 'make', 'go',
                                                                  'do']:  # Filter generic words
                    keywords.append(token.lemma_)

        return list(set(keywords))  # Return unique keywords

    def extract_keywords_tfidf(self, texts, num_keywords_per_doc=5):
        """
        Extracts top TF-IDF keywords for each document.
        This approach is more memory-intensive for large datasets if applied per document directly.
        A better approach for overall keywords or specific document keywords might be needed.
        Here, we'll focus on top keywords for each review.
        """
        if not texts:
            return [[] for _ in range(len(self.df))]

        # Use a TfidfVectorizer for keyword extraction.
        # min_df ignores terms that appear in too few documents (e.g., typos)
        # max_df ignores terms that appear in too many documents (e.g., common words)
        # stop_words='english' removes common English stop words.
        tfidf_vectorizer = TfidfVectorizer(stop_words='english', min_df=0.01, max_df=0.9)

        try:
            tfidf_matrix = tfidf_vectorizer.fit_transform(texts)
            feature_names = tfidf_vectorizer.get_feature_names_out()

            extracted_keywords_per_doc = []
            for i in range(tfidf_matrix.shape[0]):
                # Get scores for the current document
                feature_array = tfidf_matrix[i].toarray().flatten()
                # Sort indices by score in descending order
                top_n_indices = feature_array.argsort()[-num_keywords_per_doc:][::-1]
                # Get the actual keywords
                doc_keywords = [feature_names[idx] for idx in top_n_indices if feature_array[idx] > 0]
                extracted_keywords_per_doc.append(doc_keywords)
            return extracted_keywords_per_doc
        except ValueError as e:
            print(
                f"Error with TF-IDF calculation: {e}. This might happen if all documents are empty or too similar after stop_words removal.")
            return [[] for _ in range(len(self.df))]

    def assign_themes_to_reviews(self, review_text, bank, theme_mappings):
        """
        Assigns predefined themes to a review based on keywords found in its text.
        This function requires a `theme_mappings` dictionary to be defined externally.
        """
        assigned_themes = []
        text_lower = review_text.lower()

        # Get the theme mapping for the specific bank
        bank_theme_map = theme_mappings.get(bank, {})

        for theme, keywords in bank_theme_map.items():
            if any(kw.lower() in text_lower for kw in keywords):
                assigned_themes.append(theme)

        return assigned_themes if assigned_themes else ['General Feedback']  # Default theme if no specific match


# -----------------------------
# 🚀 Main Execution
# -----------------------------
if __name__ == "__main__":
    data_sources = {
        "CBE": "../data/processed/CBE_review_cleaned.csv",
        "BOA": "../data/processed/BoA_review_cleaned.csv",
        "Dashen": "../data/processed/Dashen_review_cleaned.csv"
    }

    loader = DataLoader()
    data_dict = loader.load_data(data_sources)

    for bank_key, bank_df in data_dict.items():
        processed_dfs = {}
        cleaner = DataCleaner()
        cleaned_df = cleaner.clean_data(bank_df, bank_key)
        processor = DataPreprocessor(cleaned_df)

        # 🧠 Apply VADER sentiment analysis
        cleaned_df[['sentiment', 'score', 'model_used']] = cleaned_df['review_text'].apply(
            processor.vader_sentiment_analysis
        )

        print(f"\n✅ Preprocessed Data for {bank_key}")
        print(cleaned_df[['review_text', 'sentiment', 'score']].head(10))

        # 📊 Aggregate by bank and rating
        print(f"\nAggregating sentiment by bank and rating for {bank_key}...")
        mean_sentiment_agg = processor.aggregate_sentiment_by_bank_rating()
        print(f"✅ Aggregated Sentiment for {bank_key}:")
        print(mean_sentiment_agg)

        # 🔑 Keyword Extraction using spaCy
        print(f"\nExtracting keywords using spaCy for {bank_key}...")
        cleaned_df['spacy_keywords'] = cleaned_df['review_text'].apply(processor.extract_keywords_spacy)
        print(f"✅ SpaCy Keyword Extraction Complete for {bank_key}. Sample:")
        print(cleaned_df[['review_text', 'spacy_keywords']].head())

        # 🔑 Keyword Extraction using TF-IDF
        print(f"\nExtracting keywords using TF-IDF for {bank_key}...")
        cleaned_df['tfidf_keywords'] = processor.extract_keywords_tfidf(cleaned_df['review_text'].tolist())
        print(f"✅ TF-IDF Keyword Extraction Complete for {bank_key}. Sample:")
        print(cleaned_df[['review_text', 'tfidf_keywords']].head())

        # 📝 Thematic Analysis: Manual/Rule-Based Clustering
        # Define your theme mappings based on extracted keywords.
        # This part requires human judgment and observation of the keywords.
        # EXAMPLE MAPPING (YOU WILL NEED TO REFINE THIS BASED ON YOUR ACTUAL KEYWORDS)
        bank_theme_mappings = {
            "CBE": {
                'Account Access Issues': ['login', 'otp', 'access', 'fingerprint', 'password', 'account'],
                'Transaction Performance': ['transfer', 'slow', 'fast', 'loading', 'payment', 'transaction'],
                'User Interface & Experience': ['ui', 'interface', 'design', 'bug', 'crash', 'smooth', 'easy to use'],
                'Customer Support': ['support', 'service', 'help', 'response'],
                'Feature Requests': ['feature', 'new', 'add', 'option', 'budgeting', 'bill pay']
            },
            "BOA": {
                'Account Access Issues': ['login', 'error', 'access', 'otp', 'fingerprint', 'account'],
                'Transaction Performance': ['transfer', 'slow', 'take long', 'transaction', 'payment'],
                'User Interface & Experience': ['ui', 'confusing', 'crash', 'unreliable', 'design', 'buggy'],
                'Customer Support': ['support', 'responsive', 'help', 'service'],
                'Feature Requests': ['feature', 'budgeting', 'new', 'add']
            },
            "Dashen": {
                'Account Access Issues': ['login', 'access', 'no issues', 'password'],
                'Transaction Performance': ['transfer', 'slow', 'smooth', 'balance', 'transaction'],
                'User Interface & Experience': ['ui', 'old-fashioned', 'functional', 'crash', 'optimization',
                                                'interface'],
                'Customer Support': ['support', 'excellent', 'helpful', 'team'],
                'Feature Requests': ['feature', 'bill pay', 'new', 'add']
            }
        }

        print(f"\nAssigning themes to reviews for {bank_key}...")
        cleaned_df['identified_themes'] = cleaned_df.apply(
            lambda row: processor.assign_themes_to_reviews(row['review_text'], row['bank_name'], bank_theme_mappings),
            axis=1
        )
        print(f"✅ Thematic Analysis Complete for {bank_key}. Sample:")
        print(cleaned_df[['review_text', 'spacy_keywords', 'tfidf_keywords', 'identified_themes']].head())

        processed_dfs[bank_key] = cleaned_df

        # Optionally, save the processed data to a new CSV file
        output_path = f"../data/processed/{bank_key}_reviews_processed_for_themes.csv"
        # Select and reorder columns as specified (review id, review text, sentiment label, sentiment score, identified theme(s))
        # Assuming 'review_id' could be just the DataFrame index if not explicitly present
        final_columns = ['bank_name', 'review_text', 'date', 'sentiment', 'score', 'identified_themes', 'rating',
                         'spacy_keywords', 'tfidf_keywords']
        cleaned_df[final_columns].to_csv(output_path, index=False)
        print(f"💾 Processed data for {bank_key} saved to {output_path}")

        print("\n--- All Banks Processed ---")

