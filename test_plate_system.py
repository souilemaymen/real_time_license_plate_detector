import cv2
from ultralytics import YOLO
import easyocr
import re

class PlateRecognizer:
    def __init__(self, languages=['en']):
        self.reader = easyocr.Reader(languages)
    def extract_text(self, image):
        # Appliquer un prétraitement simple
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # EasyOCR accepte aussi les images couleur, mais le gris est souvent mieux
        results = self.reader.readtext(gray, detail=0)
        text = ' '.join(results)
        cleaned_text = re.sub(r'[^A-Za-z0-9]', '', text)
        return cleaned_text

class LicensePlateSystem:
    def __init__(self, model_path='license_plate_detector.pt'):  # par défaut modèle générique
        self.detector = YOLO(model_path)
        self.recognizer = PlateRecognizer()
        self.conf_threshold = 0.5
    def process_image(self, image_path, output_path='result.jpg'):
        img = cv2.imread(image_path)
        if img is None:
            print("Erreur : impossible de lire l'image")
            return
        results = self.detector(img, conf=self.conf_threshold)
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                plate_img = img[y1:y2, x1:x2]
                if plate_img.size != 0:
                    plate_text = self.recognizer.extract_text(plate_img)
                    print(f"Plaque détectée : {plate_text}")
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0,255,0), 2)
                    cv2.putText(img, plate_text, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)
        cv2.imwrite(output_path, img)
        print(f"Résultat sauvegardé dans {output_path}")

if __name__ == '__main__':
    # Remplacez 'plaque.jpg' par le chemin de votre image test
    system = LicensePlateSystem()
    system.process_image('plaque.jpg', 'output.jpg')