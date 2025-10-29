import pandas as pd

# Combina los datos batch + velocidad para generar recomendaciones
def get_recommendations(user_id):
    df = pd.read_csv("ratings.csv")
    lugares = df["lugar"].unique()
    user_data = df[df["user_id"] == user_id]

    if user_data.empty:
        return lugares[:3].tolist()
    else:
        rated = user_data["lugar"].unique()
        not_rated = [l for l in lugares if l not in rated]
        return not_rated[:3]
