import cv2
from ultralytics import YOLO
import easyocr
import re


class PlateRecognizer:
    def __init__(self, languages=['en']):
        self.reader = easyocr.Reader(languages)

    def extract_text(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        results = self.reader.readtext(gray, detail=0)
        text = ' '.join(results)
        return re.sub(r'[^A-Za-z0-9]', '', text)


class LicensePlateDetector:
    def __init__(self, model_path, conf_threshold=0.5):
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
        self.recognizer = PlateRecognizer()

    def detect_and_recognize(self, image):
        """Retourne (image_annotée, liste_des_textes) pour une image BGR."""
        results = self.model(image, conf=self.conf_threshold)
        texts = []
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                plate_crop = image[y1:y2, x1:x2]
                if plate_crop.size != 0:
                    text = self.recognizer.extract_text(plate_crop)
                    if text:
                        texts.append(text)
                    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(image, text, (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        return image, texts