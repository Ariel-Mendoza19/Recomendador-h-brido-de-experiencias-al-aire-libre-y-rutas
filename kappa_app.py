import streamlit as st
from stream_processor import update_recommendations
from data_stream import generate_stream, append_to_csv
import threading

st.set_page_config(page_title="Arquitectura Kappa - Recomendador", page_icon="⚡", layout="centered")

st.title("⚡ Recomendador Turístico - Arquitectura Kappa")
st.write("Procesamiento en tiempo real con flujo de datos continuo")

user_id = st.text_input("🧑 Ingrese su ID de usuario (por ejemplo: user1)")

if st.button("Ver recomendaciones"):
    recs = update_recommendations(user_id)
    st.write("### Lugares recomendados en tiempo real:")
    for r in recs:
        st.write(f"🏞️ {r}")

# ---- Iniciar flujo en segundo plano ----
if st.button("Iniciar flujo de datos simulados"):
    st.info("El flujo de datos comenzó... se agregarán nuevas valoraciones cada 2 segundos.")

    def start_stream():
        for data in generate_stream():
            append_to_csv(data)

    thread = threading.Thread(target=start_stream, daemon=True)
    thread.start()
