import pandas as pd
from load_data import DataLoader
import os


class DataCleaner:

    def __init__(self, output_dir: str = "../data/processed/"):
        """
        Initializes the DataCleaner class.
        """
        self.output_dir = output_dir

    def clean_data(self, df: pd.DataFrame, key) -> pd.DataFrame:

        print("\n📊 Data Description:")
        print(df.describe())

        if df.empty:
            print("The DataFrame is empty, so it has no columns.")
        else:
            print("Columns in the DataFrame:")
            print(df.columns.tolist())

        # column renaming code
        renamed_column = {
            'Review Text': "review_text",
            'Rating': 'rating',
            'Date': 'date',
            'Bank/App Name': 'bank_name',
            'Source': 'source'
        }
        df = df.rename(columns=renamed_column)

        print("Columns after renaming:", df.columns.tolist())

        # 2. Check shape and size
        print(f"\n🧱 Shape: {df.shape} (rows, columns)")
        print(f"📦 Size: {df.size} (total elements)")

        # Check Info
        print(f"\n🧱 Data Types of the df: {df.dtypes}")
        print(f"\n🧱 Info of the DF: {df.info}")
        print(f"\n🧱 Duplicated Values: {df.duplicated().sum()}")

        # Check for missing values
        print("\n🕳️ Missing Values Per Column:")
        print(df.isnull().sum())

        # Drop completely empty rows and columns
        df = df.dropna(how='all')  # Drop rows where all values are NaN
        df = df.dropna(axis=1, how='all')  # Drop columns where all values are NaN

        # # 5. Optional: Fill missing values with placeholders or drop them
        # # df = df.fillna("Unknown")  # or choose specific columns
        #
        # # 6. Strip whitespace from string columns
        # str_cols = df.select_dtypes(include='object').columns
        # df[str_cols] = df[str_cols].apply(lambda x: x.str.strip())

        #  Normalize dates to YYYY-MM-DD
        df['date'] = pd.to_datetime(df['date'])
        df['date'] = df['date'].dt.strftime('%Y-%m-%d')
        print(f" Normalized 'at' column to 'date' (YYYY-MM-DD).{df["date"]}")

        print("\n✅ Data cleaning complete.")
        file_name = key+"_review_cleaned.csv"
        file_path = os.path.join(self.output_dir, file_name)
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
                print(f"🔄 Existing file '{file_name}' found and removed.")
            else:
                print(f"🆕 No existing file '{file_name}' found. Creating new file.")
            df.to_csv(file_path)
            print(f"✅ DataFrame successfully saved to: {file_path}")
        except Exception as e:
            print(f"❌ Error saving DataFrame to {file_path}: {e}")
        return df



if __name__ == "__main__":
    df = {
        "CBE": "../data/raw/Commercial Bank of Ethiopia_reviews.csv",
        "BOA": "../data/raw/BoA Mobile_reviews.csv",
        "Dashen": "../data/raw/Dashen Bank_reviews.csv"
    }
    load_data = DataLoader()
    all_data = load_data.load_data(df)
    clean_data = DataCleaner()
    for key, data in all_data.items():
        cd = clean_data.clean_data(data, key)
        print(cd)
