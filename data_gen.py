import pandas as pd
import numpy as np
from pathlib import Path

Path("data").mkdir(exist_ok=True)

# Generar 50 items (experiencias al aire libre)
items = []
for i in range(50):
    items.append({
        "item_id": f"it{i+1}",
        "title": f"Ruta o Lugar {i+1}",
        "type": np.random.choice(["sendero","mirador","parque","ruta cultural","actividad"]),
        "city_dist_km": round(np.random.uniform(0.5, 40.0),1)
    })
items_df = pd.DataFrame(items)
items_df.to_csv("data/items.csv", index=False)

# Generar 500 usuarios y 3000 valoraciones aleatorias
user_ids = [f"user{u+1}" for u in range(500)]
ratings = []
for _ in range(3000):
    u = np.random.choice(user_ids)
    it = np.random.choice(items_df["item_id"])
    rating = np.random.randint(1,6)
    ts = np.random.randint(1609459200, 1735689600)  # años 2021-2024
    ratings.append({"user_id": u, "item_id": it, "rating": rating, "timestamp": ts})
ratings_df = pd.DataFrame(ratings)
ratings_df.to_csv("data/sample_ratings.csv", index=False)

print("✅ Generados: data/items.csv y data/sample_ratings.csv")
