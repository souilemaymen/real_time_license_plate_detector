import streamlit as st
import cv2
import numpy as np
import re
from ultralytics import YOLO
import easyocr
import av
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase

# ------------------ Configuration de la page ------------------
st.set_page_config(page_title="License Plate Detector - Realtime", layout="wide")
st.title("🚗 Détection et reconnaissance de plaques en temps réel")
st.markdown("Placez votre plaque devant la webcam, la détection se fait automatiquement.")


# ------------------ Chargement des modèles (mis en cache) ------------------
@st.cache_resource
def load_model(model_path):
    return YOLO(model_path)


@st.cache_resource
def load_ocr(languages=['en']):
    return easyocr.Reader(languages)


# ------------------ Paramètres dans la sidebar ------------------
with st.sidebar:
    st.header("⚙️ Paramètres")
    model_path = st.text_input("Chemin du modèle YOLO", value="license_plate_detector.pt")
    conf_thresh = st.slider("Seuil de confiance", 0.25, 0.9, 0.5, 0.05)
    st.markdown("---")
    st.info("Assurez-vous que le modèle détecte les plaques d'immatriculation.")

# Chargement effectif
model = load_model(model_path)
ocr = load_ocr(['en'])


# ------------------ Fonction de nettoyage du texte ------------------
def extract_plate_text(plate_img):
    """Extrait le texte d'une image de plaque (numpy array BGR)"""
    gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
    results = ocr.readtext(gray, detail=0)
    text = ' '.join(results)
    # Garder seulement lettres majuscules et chiffres
    cleaned = re.sub(r'[^A-Z0-9]', '', text.upper())
    return cleaned


# ------------------ Processeur vidéo pour streamlit-webrtc ------------------
class PlateProcessor(VideoProcessorBase):
    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        # 1. Convertir le frame PyAV en tableau NumPy (format BGR)
        img = frame.to_ndarray(format="bgr24")

        # 2. Détection YOLO sur l'image
        results = model(img, conf=conf_thresh)

        # 3. Pour chaque détection, appliquer OCR et annoter
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                plate_crop = img[y1:y2, x1:x2]
                if plate_crop.size != 0:
                    plate_text = extract_plate_text(plate_crop)
                    # Dessiner rectangle et texte
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(img, plate_text, (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        # 4. Retourner le frame annoté
        return av.VideoFrame.from_ndarray(img, format="bgr24")


# ------------------ Lancement du flux vidéo ------------------
st.markdown("### 📸 Flux webcam en direct")
ctx = webrtc_streamer(
    key="license-plate-detection",
    video_processor_factory=PlateProcessor,  # Nouvelle API
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
)

if ctx.video_processor:
    st.success("✅ Caméra active – la détection s'affiche en direct.")
else:
    st.info("🔴 Cliquez sur 'START' pour activer la webcam.")