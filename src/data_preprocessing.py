import re
import emoji
from langdetect import detect, DetectorFactory
from googletrans import Translator

class DataProprocessor:
    def __init__(self, df:pd.DataFrame):
        self.df = df

    # Set seed for reproducibility in langdetect
    DetectorFactory.seed = 0

    def clean_text(text):
        """Basic text cleaning: remove URLs, mentions, hashtags, and extra spaces."""
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        text = re.sub(r'@\w+', '', text)
        text = re.sub(r'#\w+', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def replace_emojis_with_text(text):
        """Replaces emojis with their textual description."""
        return emoji.demojize(text, delimiters=(" ", " "))

    def detect_language(text):
        """Detects the language of the text."""
        try:
            return detect(text)
        except:
            return 'unknown' # Handle cases where language detection might fail

    # Apply preprocessing
    df['cleaned_text'] = df['review_text'].apply(clean_text)
    df['emoji_text_replaced'] = df['cleaned_text'].apply(replace_emojis_with_text)
    df['language'] = df['cleaned_text'].apply(detect_language)


    translator = Translator()

    def translate_amharic_to_english(text, lang):
        if lang == 'am':
            try:

                time.sleep(0.1)
                return translator.translate(text, src='am', dest='en').text
            except Exception as e:
                print(f"Translation error for '{text}': {e}")
                return text # Return original text if translation fails
        return text

# Apply translation to Amharic texts
df['translated_text'] = df.apply(
    lambda row: translate_amharic_to_english(row['emoji_text_replaced'], row['language']),
    axis=1
)

print("\n--- Preprocessed Data ---")
print(df[['text', 'language', 'emoji_text_replaced', 'translated_text']].head())