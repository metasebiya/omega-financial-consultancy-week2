from google_play_scraper import reviews, app
import pandas as pd
from datetime import datetime
import time


def scrape_data(package_names, target_review_count):
    for pkg_name, pkg_id in package_names.items():
        print(f"package_name:{pkg_name}:package_id:{pkg_id}")
        # 📦 Get app name
        app_info = app(pkg_id)
        app_name = app_info['title']

        # 🧺 Initialize storage
        all_reviews = []
        token = None
        # 🔁 Keep fetching in batches until target is reached
        while len(all_reviews) < target_review_count:
            count_to_fetch = min(200, target_review_count - len(all_reviews))  # Max 200 per call

            reviews_batch, token = reviews(
                pkg_id,
                lang='en',
                country='us',
                count=count_to_fetch,
                continuation_token=token  # Token from previous batch
            )

            if not reviews_batch:
                print("No more reviews found.")
                break

            all_reviews.extend(reviews_batch)

            print(f"Fetched {len(all_reviews)} / {target_review_count}")

            time.sleep(1)  # Politeness delay (important to avoid getting blocked)

        # 📃 Format for export
        data = [{
            "Review Text": r['content'],
            "Rating": r['score'],
            "Date": r['at'].strftime('%Y-%m-%d %H:%M:%S'),
            "Bank/App Name": app_name,
            "Source": "Google Play"
        } for r in all_reviews]

        # 📊 Create DataFrame and export
        df = pd.DataFrame(data)
        df.to_csv(f"../data/raw/{app_name}_reviews.csv", index=False)

    print("✅ Scraping complete! Saved to CSV.")

if __name__ == "__main__":
    # 🎯 App package name and target review count
    package_names = {
        "CBE": "com.combanketh.mobilebanking",
        "BOA": "com.boa.boaMobileBanking",
        "Dashen": "com.dashen.dashensuperapp"
    }
    target_review_count = 2000
    scrape_data(package_names, target_review_count)