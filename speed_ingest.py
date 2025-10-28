import argparse
import pandas as pd
from pathlib import Path
import time

RECENT_FILE = Path("data/recent_ratings.csv")
RECENT_FILE.parent.mkdir(exist_ok=True)

def append_rating(user_id, item_id, rating, timestamp=None):
    if timestamp is None:
        timestamp = int(time.time())
    row = {"user_id": user_id, "item_id": item_id, "rating": rating, "timestamp": timestamp}
    if RECENT_FILE.exists():
        df = pd.read_csv(RECENT_FILE)
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    else:
        df = pd.DataFrame([row])
    df = df.sort_values("timestamp").tail(200)  # mantener últimas 200
    df.to_csv(RECENT_FILE, index=False)
    print("🆕 Valoración añadida:", row)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--add", nargs=3, metavar=('USER','ITEM','RATING'), help="Agregar rating: USER ITEM RATING")
    args = parser.parse_args()
    if args.add:
        user, item, rating = args.add
        append_rating(user, item, int(rating))
