import time
import numpy as np
import cv2
import streamlit as st
import pandas as pd
from datetime import datetime
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration, VideoTransformerBase
from tensorflow.keras.models import load_model
from database import init_db, registrar_acceso

# ---------------- CONFIGURACIÓN INICIAL ----------------
st.set_page_config(page_title="Clasificador en vivo", page_icon="🎥", layout="wide")
st.title("🎥 Clasificación en vivo con control de acceso (Big Data)")

# Inicializar base de datos
init_db()

# ---------------- CONTROL DE ACCESO ----------------
st.sidebar.header("🎓 Acceso al sistema")
nombre = st.sidebar.text_input("Ingrese su nombre:")
carrera = st.sidebar.selectbox(
    "Seleccione su carrera:",
    ["Seleccione...", "Big Data", "Otra carrera"]
)

if carrera != "Big Data":
    st.warning("🚫 Acceso permitido solo para estudiantes de la carrera de **Big Data**.")
    st.stop()
else:
    if nombre:
        registrar_acceso(nombre, carrera)
        st.success(f"✅ Acceso permitido. Bienvenido/a {nombre} a la plataforma de entrenamiento.")
    else:
        st.info("Por favor, ingrese su nombre para continuar.")
        st.stop()

# ---------------- CARGA DEL MODELO ----------------
MODEL_PATH = "keras_Model.h5"
LABELS_PATH = "labels.txt"

@st.cache_resource
def load_model_cached(model_path: str):
    return load_model(model_path, compile=False)

@st.cache_data
def load_labels(labels_path: str):
    with open(labels_path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f.readlines()]

try:
    model = load_model_cached(MODEL_PATH)
    labels = load_labels(LABELS_PATH)
except Exception as e:
    st.error(f"No se pudo cargar el modelo o etiquetas: {e}")
    st.stop()

# ---------------- AJUSTES DE CÁMARA ----------------
st.sidebar.header("🎥 Ajustes de cámara")
facing = st.sidebar.selectbox("Tipo de cámara", ["auto", "user (frontal)", "environment (trasera)"], index=0)
quality = st.sidebar.selectbox("Calidad de video", ["640x480", "1280x720", "1920x1080"], index=1)
w, h = map(int, quality.split("x"))
video_constraints = {"width": w, "height": h}
if facing != "auto":
    video_constraints["facingMode"] = facing.split(" ")[0]
media_constraints = {"video": video_constraints, "audio": False}

RTC_CONFIGURATION = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})

# ---------------- TRANSFORMADOR DE VIDEO ----------------
class VideoTransformer(VideoTransformerBase):
    def __init__(self) -> None:
        self.latest = {"class": None, "confidence": 0.0}
        self.model = model
        self.labels = labels

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        resized = cv2.resize(img, (224, 224), interpolation=cv2.INTER_AREA)
        x = (resized.astype(np.float32).reshape(1, 224, 224, 3) / 127.5) - 1.0

        pred = self.model.predict(x, verbose=0)
        idx = int(np.argmax(pred))
        label = self.labels[idx] if idx < len(self.labels) else f"Clase {idx}"
        conf = float(pred[0][idx])
        self.latest = {"class": label, "confidence": conf}

        overlay = img.copy()
        text = f"{label} | {conf*100:.1f}%"
        cv2.rectangle(overlay, (5, 5), (5 + 8*len(text), 45), (0, 0, 0), -1)
        cv2.putText(overlay, text, (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, cv2.LINE_AA)
        return overlay

# ---------------- INTERFAZ PRINCIPAL ----------------
left, right = st.columns([2, 1], gap="large")

with left:
    st.subheader("Cámara en vivo")
    webrtc_ctx = webrtc_streamer(
        key="keras-live",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=RTC_CONFIGURATION,
        media_stream_constraints=media_constraints,
        video_transformer_factory=VideoTransformer,
        async_processing=True,
    )

with right:
    st.subheader("Resultados")
    result_placeholder = st.empty()
    progress_placeholder = st.empty()

    if webrtc_ctx and webrtc_ctx.state.playing:
        for _ in range(300000):
            if not webrtc_ctx.state.playing:
                break
            vt = webrtc_ctx.video_transformer
            if vt and vt.latest["class"] is not None:
                cls = vt.latest["class"]
                conf = vt.latest["confidence"]
                result_placeholder.markdown(f"**Clase detectada:** `{cls}`\n\n**Confianza:** `{conf*100:.2f}%`")
                progress_placeholder.progress(min(max(conf, 0.0), 1.0))
            time.sleep(0.3)
    else:
        st.info("Activa la cámara para ver las predicciones en vivo.")

# ---------------- MODO ALTERNATIVO ----------------
st.markdown("---")
with st.expander("📸 Modo alternativo (foto puntual)"):
    snap = st.camera_input("Captura una imagen")
    if snap is not None:
        file_bytes = np.asarray(bytearray(snap.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, 1)
        resized = cv2.resize(img, (224, 224))
        x = (resized.astype(np.float32).reshape(1, 224, 224, 3) / 127.5) - 1.0
        pred = model.predict(x, verbose=0)
        idx = int(np.argmax(pred))
        label = labels[idx]
        conf = float(pred[0][idx])
        st.image(img, caption=f"{label} | {conf*100:.2f}%")
        st.success(f"Predicción: **{label}** ({conf*100:.2f}%)")

