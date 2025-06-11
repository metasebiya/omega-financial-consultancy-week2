import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import os
from datetime import datetime
from collections import Counter  # Make sure this is imported for generate_insights_report

# --- Define the output directory for reports and figures ---
REPORTS_DIR = '../reports/'
FIGURES_DIR = os.path.join(REPORTS_DIR, 'figures')
os.makedirs(REPORTS_DIR, exist_ok=True)  # Ensure the main reports directory exists
os.makedirs(FIGURES_DIR, exist_ok=True)  # Ensure the figures subdirectory exists

# Set style for visualizations
sns.set_style("whitegrid")
sns.set_palette("husl")


class Visualizer:

    def __init__(self, df: pd.DataFrame):
        self.data = df.copy()  # Use .copy() to avoid SettingWithCopyWarning later

        # Ensure 'review_date' is datetime type
        if 'date' in self.data.columns and not pd.api.types.is_datetime64_any_dtype(self.data['date']):
            self.data['date'] = pd.to_datetime(self.data['date'], errors='coerce')

        # Ensure list-like columns are actual lists, not string representations of lists
        list_cols = ['identified_themes', 'spacy_keywords', 'tfidf_keywords']
        for col in list_cols:
            if col in self.data.columns:
                # If column contains strings that look like lists, convert them
                if self.data[col].apply(lambda x: isinstance(x, str) and x.startswith('[') and x.endswith(']')).any():
                    import ast
                    self.data[col] = self.data[col].apply(
                        lambda x: ast.literal_eval(x) if isinstance(x, str) and x.startswith('[') else []
                    )
                # Ensure values are lists; if not, convert to empty list (e.g., if NaN or None)
                self.data[col] = self.data[col].apply(lambda x: x if isinstance(x, list) else [])

    # Function to generate sentiment trend plot
    def plot_sentiment_trends(self):
        print("Generating Sentiment Trends Over Time plot...")
        plot_df = self.data.dropna(subset=['date', 'sentiment', 'bank_name']).copy()
        if plot_df.empty:
            print("Not enough data to plot sentiment trends after dropping NaNs.")
            return

        for bank in plot_df['bank_name'].unique():
            bank_data = plot_df[plot_df['bank_name'] == bank]
            sentiment_map = {'positive': 1, 'neutral': 0, 'negative': -1}
            bank_data['sentiment_score'] = bank_data['sentiment'].map(sentiment_map)

            monthly_sentiment = bank_data.groupby(bank_data['date'].dt.to_period('M'))['sentiment_score'].mean()
            monthly_sentiment.index = monthly_sentiment.index.to_timestamp()

            plt.figure(figsize=(12, 7))
            plt.plot(monthly_sentiment.index, monthly_sentiment.values, label=bank, marker='o', linestyle='-')
            plt.title(f'Sentiment Trends Over Time - {bank}', fontsize=16, pad=20)
            plt.xlabel('Date', fontsize=12)
            plt.ylabel('Average Sentiment Score', fontsize=12)
            plt.legend(title='Bank')
            plt.xticks(rotation=45)
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.tight_layout()

            safe_bank_name = bank.lower().replace(" ", "_").replace("/", "_").replace("\\", "_")
            file_path = os.path.join(FIGURES_DIR, f'sentiment_trends_{safe_bank_name}.png')
            plt.savefig(file_path)
            plt.show()
            plt.close()
            print(f"Sentiment Trends plot saved for {bank}: {file_path}")

    def plot_rating_distribution(self):
        print("Generating Rating Distribution by Bank plot...")
        plot_df = self.data.dropna(subset=['rating', 'bank_name']).copy()
        if plot_df.empty:
            print("Not enough data to plot rating distribution after dropping NaNs.")
            return

        for bank in plot_df['bank_name'].unique():
            bank_data = plot_df[plot_df['bank_name'] == bank]

            plt.figure(figsize=(10, 6))
            sns.countplot(data=bank_data, x='rating', palette='viridis')
            plt.title(f'Rating Distribution - {bank}', fontsize=16, pad=20)
            plt.xlabel('Rating (1-5 Stars)', fontsize=12)
            plt.ylabel('Number of Reviews', fontsize=12)
            plt.tight_layout()

            safe_bank_name = bank.lower().replace(" ", "_").replace("/", "_").replace("\\", "_")
            file_path = os.path.join(FIGURES_DIR, f'rating_distribution_{safe_bank_name}.png')
            plt.savefig(file_path)
            plt.show()
            plt.close()
            print(f"Rating Distribution plot saved for {bank}: {file_path}")

    # Function to generate keyword cloud per bank
    def plot_keyword_cloud(self):
        print("Generating Keyword Clouds per Bank...")
        plot_df = self.data.dropna(subset=['bank_name', 'identified_themes']).copy()

        if plot_df.empty:
            print("Not enough data to plot keyword clouds after dropping NaNs.")
            return

        for bank in plot_df['bank_name'].unique():
            bank_data = plot_df[plot_df['bank_name'] == bank]
            # Flatten the list of lists into a single list of themes
            all_themes_flat = [theme for sublist in bank_data['identified_themes'] if isinstance(sublist, list) for
                               theme in sublist]
            text = ' '.join(all_themes_flat)  # Join all themes into a single string

            if text.strip():  # Check if there is any text to generate a word cloud
                try:
                    wordcloud = WordCloud(width=800, height=400, background_color='white', max_words=50).generate(text)
                    plt.figure(figsize=(10, 5))
                    plt.imshow(wordcloud, interpolation='bilinear')
                    plt.title(f'Keyword Cloud for {bank}', fontsize=16, pad=15)  # Enhanced title
                    plt.axis('off')
                    plt.tight_layout()
                    safe_bank_name = bank.lower().replace(" ", "_").replace("/", "_").replace("\\",
                                                                                              "_")  # Create a safe filename
                    plt.savefig(os.path.join(FIGURES_DIR, f'keyword_cloud_{safe_bank_name}.png'))  # Save to FIGURES_DIR
                    plt.show()
                    plt.close()
                    print(f"Keyword Cloud for {bank} saved and displayed.")
                except NameError:
                    print(
                        f"Skipping Word Cloud for {bank}: 'wordcloud' library not installed. Install with 'pip install wordcloud'.")
                except Exception as e:
                    print(f"Error generating Word Cloud for {bank}: {e}")
            else:
                print(f"No valid theme text to generate Word Cloud for {bank}.")
        print("Finished generating all Keyword Clouds.")

    # Function to derive insights and recommendations
    def generate_insights_report(self):
        print("Generating Insights Report...")
        insights = []

        if self.data.empty:
            print("Cannot generate insights report: Data is empty.")
            return "No data available for report."

        for bank in self.data['bank_name'].unique():
            bank_data = self.data[self.data['bank_name'] == bank]

            # Drivers: High sentiment scores and positive themes
            positive_reviews = bank_data[bank_data['sentiment'].astype(str).str.lower() == 'positive']
            # Flatten the list of lists for identified_themes before counting
            all_positive_themes = [theme for sublist in positive_reviews['identified_themes'] if
                                   isinstance(sublist, list) for theme in sublist]
            driver_themes = Counter(all_positive_themes).most_common(2)

            drivers = []
            if driver_themes:
                for theme, count in driver_themes:
                    drivers.append(f"**{bank}**: Positive feedback on **'{theme}'** (mentioned {count} times)")
            else:
                drivers.append(f"**{bank}**: No clear drivers identified from positive reviews.")

            # Pain points: Low ratings and negative themes
            negative_reviews = bank_data[bank_data['sentiment'].astype(str).str.lower() == 'negative']
            # Flatten the list of lists for identified_themes before counting
            all_negative_themes = [theme for sublist in negative_reviews['identified_themes'] if
                                   isinstance(sublist, list) for theme in sublist]
            pain_themes = Counter(all_negative_themes).most_common(2)

            pain_points = []
            if pain_themes:
                for theme, count in pain_themes:
                    pain_points.append(f"**{bank}**: Complaints about **'{theme}'** (mentioned {count} times)")
            else:
                pain_points.append(f"**{bank}**: No clear pain points identified from negative reviews.")

            # Recommendations based on themes
            recommendations = []
            themes_in_negative_reviews_flat = [theme for sublist in negative_reviews['identified_themes'] if
                                               isinstance(sublist, list) for theme in sublist]

            # Using specific themes that might appear in analysis
            if 'Transaction Performance' in themes_in_negative_reviews_flat:
                recommendations.append(
                    f"**{bank}**: Optimize app performance for faster loading and smoother transactions.")
            if 'Account Access Issues' in themes_in_negative_reviews_flat:
                recommendations.append(
                    f"**{bank}**: Enhance login reliability and consider integrating AI chatbot for faster issue resolution.")

            # Check for generic "Feature Requests" among all themes for the bank
            if 'Feature Requests' in [theme for sublist in bank_data['identified_themes'] if
                                      isinstance(sublist, list) for theme in sublist]:
                recommendations.append(
                    f"**{bank}**: Analyze and prioritize highly requested features (e.g., fingerprint login, budgeting tools).")
            if not recommendations:
                recommendations.append(
                    f"**{bank}**: Further qualitative review is recommended to identify specific improvement areas.")

            insights.append(f"### {bank} Insights ###\n")
            insights.append("#### Drivers ####")
            insights.extend([f"- {d}" for d in drivers])
            insights.append("#### Pain Points ####")
            insights.extend([f"- {p}" for p in pain_points])
            insights.append("#### Recommendations ####")
            insights.extend([f"- {r}" for r in recommendations])
            insights.append("\n")

        print("\n### Comparative Analysis of 'Transaction Performance' issues ###")
        perf_issue_summary = []
        for bank in self.data['bank_name'].unique():
            bank_negative_perf = self.data[
                (self.data['bank_name'] == bank) &
                (self.data['sentiment'].astype(str).str.lower() == 'negative') &
                (self.data['identified_themes'].apply(
                    lambda x: 'Transaction Performance' in x if isinstance(x, list) else False))
                ]
            perf_issue_summary.append(
                f"**{bank}**: {len(bank_negative_perf)} negative reviews mentioning 'Transaction Performance'.")
            if not bank_negative_perf.empty:
                print(
                    f"Sample review for {bank} 'Transaction Performance' issue: '{bank_negative_perf.sample(1)['review_text'].iloc[0]}'")

        insights.append("### Comparative Analysis ###")
        insights.append("#### 'Transaction Performance' Issues Across Banks ####")
        insights.extend([f"- {summary}" for summary in perf_issue_summary])
        insights.append("\n")

        report = f"# BSW2: Customer Experience Analytics Report\n\n"
        report += f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        report += "## Executive Summary\n\n"
        report += "This report analyzes customer feedback for Ethiopian banking apps (CBE, Bank of Abyssinia, Dashen Bank) to identify key satisfaction drivers, pain points, and actionable recommendations for app improvement.\n\n"
        report += "## Detailed Insights and Recommendations\n\n"
        report += "\n".join(insights)

        report += "\n## Visualizations\n"
        report += f"- **Sentiment Trends**: See '{os.path.join(FIGURES_DIR, 'sentiment_trends.png')}' for monthly sentiment trends per bank.\n"
        report += f"- **Rating Distribution**: See '{os.path.join(FIGURES_DIR, 'rating_distribution.png')}' for rating distributions.\n"
        for bank in self.data['bank_name'].unique():
            safe_bank_name = bank.lower().replace(" ", "_").replace("/", "_").replace("\\", "_")
            report += f"- **Keyword Cloud for {bank}**: See '{os.path.join(FIGURES_DIR, f'keyword_cloud_{safe_bank_name}.png')}'.\n"
        report += "\n"

        report += "## Ethical Considerations and Biases\n\n"
        report += "It's important to acknowledge potential biases in the user review data:\n"
        report += "- **Self-Selection Bias (Negative Skew)**: Users with exceptionally positive or negative experiences are more likely to leave reviews, potentially overrepresenting extreme sentiments compared to the average user base.\n"
        report += "- **Extremity Bias**: Reviews often gravitate towards 1-star or 5-star ratings, with fewer reviews in the middle range, which can skew perceived average satisfaction.\n"
        report += "- **Temporal Bias**: Recent app updates or public events might significantly influence current review sentiments, which may not reflect long-term user satisfaction.\n"
        report += "These biases should be considered when interpreting the insights and formulating strategies.\n"

        # --- CORRECTED REPORT SAVING PATH ---
        report_file_path = os.path.join(REPORTS_DIR, 'insights_report.md')
        with open(report_file_path, 'w', encoding='utf-8') as f:
            f.write(report)
        print(f"Insights Report saved to: {report_file_path}")
        return report
