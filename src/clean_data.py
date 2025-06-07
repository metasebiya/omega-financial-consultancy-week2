import pandas as pd
from load_data import DataLoader
import os


class DataCleaner:

    def __init__(self, output_dir: str = "../data/processed/"):
        """
        Initializes the DataCleaner class with an output directory.
        """
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)  # Ensure directory exists

    def clean_data(self, df: pd.DataFrame, key: str) -> pd.DataFrame:
        if df.empty:
            print("\n⚠️ The DataFrame is empty.")
            return df

        print("\n📊 Initial Data Description:")
        print(df.describe(include='all'))
        print("\n📋 Initial Columns:", df.columns.tolist())

        # Column renaming
        rename_map = {
            'Review Text': "review_text",
            'Rating': 'rating',
            'Date': 'date',
            'Bank/App Name': 'bank_name',
            'Source': 'source'
        }
        df.rename(columns=rename_map, inplace=True)
        print("✅ Columns renamed to:", df.columns.tolist())

        # Shape and basic info
        print(f"\n🧱 Shape: {df.shape}")
        print(f"📦 Total Elements: {df.size}")
        print(f"\n📂 Data Types:\n{df.dtypes}")
        print("\n🧾 Missing Values Per Column:")
        print(df.isnull().sum())
        print(f"\n🔍 Duplicate Rows: {df.duplicated().sum()}")

        # Drop rows and columns with all NaN values
        df.dropna(how='all', inplace=True)
        df.dropna(axis=1, how='all', inplace=True)

        # Strip whitespace in string columns
        str_cols = df.select_dtypes(include='object').columns
        df[str_cols] = df[str_cols].apply(lambda col: col.str.strip())

        # Normalize date column
        if 'date' in df.columns:
            try:
                df['date'] = pd.to_datetime(df['date'], errors='coerce')
                df['date'] = df['date'].dt.strftime('%Y-%m-%d')
                print("📅 Date column normalized to YYYY-MM-DD format.")
            except Exception as e:
                print(f"❌ Error parsing dates: {e}")

        # Save cleaned data
        file_name = f"{key}_review_cleaned.csv"
        file_path = os.path.join(self.output_dir, file_name)
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
                print(f"🔄 Replacing existing file: {file_name}")
            df.to_csv(file_path, index=False)
            print(f"✅ Cleaned data saved to: {file_path}")
        except Exception as e:
            print(f"❌ Failed to save cleaned data: {e}")

        return df


if __name__ == "__main__":
    data_sources = {
        "CBE": "../data/raw/Commercial Bank of Ethiopia_reviews.csv",
        "BOA": "../data/raw/BoA Mobile_reviews.csv",
        "Dashen": "../data/raw/Dashen Bank_reviews.csv"
    }

    loader = DataLoader()
    data_dict = loader.load_data(data_sources)
    cleaner = DataCleaner()

    for bank_key, bank_df in data_dict.items():
        cleaned_df = cleaner.clean_data(bank_df, bank_key)
        print(f"\n🧼 Cleaned Data Preview for {bank_key}:")
        print(cleaned_df.head())
