import pandas as pd

# Simula la llegada de nuevas calificaciones en tiempo real
def add_new_rating(user_id, lugar, rating):
    df = pd.read_csv("ratings.csv")
    new_row = {"user_id": user_id, "lugar": lugar, "rating": rating}
    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    df.to_csv("ratings.csv", index=False)
