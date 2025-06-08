"""
load_data.py - Data loading functions for customer review data

This module contains functions to load customer review data from play store

Author: [Metasebiya Bizuneh]
Created: June 5, 2025
"""

import os
import pandas as pd


class DataLoader:
    # def __init__(self, customer_review):
    #     self.customer_review = customer_review

    def load_data(self, customer_review:pd.DataFrame):
        """
        Load a CSV file into a pandas DataFrame for financial analysis

        Parameters:
            file_path (str): Path to the CSV file (e.g., 'data/raw/{app_name}.csv')

        Returns:
            pd.DataFrame: Loaded customer review data as a DataFrame

        Raises:
            FileNotFoundError: If the specified file does not exist
        """
        all_data_files = {}
        for pkg_name, pkg_path in customer_review.items():
            if not os.path.isfile(pkg_path):
                raise FileNotFoundError(f"File {pkg_path} does not exist")
            df = pd.read_csv(pkg_path)
            all_data_files[pkg_name] = df
        return all_data_files

if __name__ == "__main__":
    df = {
        "CBE": "../data/raw/Commercial Bank of Ethiopia_reviews.csv",
        "BOA": "../data/raw/BoA Mobile_reviews.csv",
        "Dashen": "../data/raw/Dashen Bank_reviews.csv"
    }
    data = DataLoader()
    all_data = data.load_data(df)
    print(all_data)