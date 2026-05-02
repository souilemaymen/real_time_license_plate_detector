# 🚗 License Plate Detection & Recognition System

Un système complet de détection et reconnaissance de plaques d'immatriculation utilisant **YOLO** pour la localisation et **EasyOCR** pour la lecture du texte.  
L'interface web est construite avec **Streamlit**, supportant l'upload d'image et le flux vidéo en temps réel via WebRTC.

## ✨ Fonctionnalités

- 📁 **Upload d'image** : chargez une photo, détection + OCR s'affichent en quelques secondes.
- 🎥 **Flux vidéo en direct** : activez votre webcam, la détection se fait en temps réel sur chaque frame.
- 🖨️ **Affichage clair** : les plaques sont encadrées en vert avec le texte reconnu.
- 📋 **Logs** : liste des plaques détectées avec timestamp.
- ⚙️ **Paramètres ajustables** : seuil de confiance YOLO, chemin du modèle personnalisé.
- 📥 **Téléchargement** : sauvegardez l'image annotée (mode image).

## 🧰 Prérequis

- Python 3.8 ou supérieur
- Une webcam (pour le mode temps réel)
- Un modèle YOLO entraîné pour les plaques (ex: `yolov8n-license-plate.pt`)

## 🔧 Installation

1. **Clonez ou téléchargez** ce dépôt.

2. **Créez un environnement virtuel** (recommandé) :
   ```bash
   python -m venv venv
   source venv/bin/activate   # Linux/Mac
   venv\Scripts\activate      # Windows
   pip install -r requirements.txt
   streamlit run streamlit_app




