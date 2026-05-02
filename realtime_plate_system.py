import cv2
from ultralytics import YOLO
import easyocr
import re
import time


class PlateRecognizer:
    def __init__(self, languages=['en']):
        self.reader = easyocr.Reader(languages)

    def extract_text(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        results = self.reader.readtext(gray, detail=0)
        text = ' '.join(results)
        cleaned_text = re.sub(r'[^A-Za-z0-9]', '', text)
        return cleaned_text


class RealtimePlateSystem:
    def __init__(self, model_path='yolov8n-license-plate.pt', conf_threshold=0.5):
        self.detector = YOLO(model_path)
        self.recognizer = PlateRecognizer()
        self.conf_threshold = conf_threshold

    def run(self, source=0):
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            print("Erreur : impossible d'ouvrir la source vidéo")
            return

        # Pour sauvegarder la vidéo (optionnel)
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter('output_realtime.mp4', fourcc, 20.0, (640, 480))

        frame_count = 0
        skip_frames = 2  # Analyse OCR toutes les 2 images pour fluidité

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            start_time = time.time()

            # Détection YOLO
            results = self.detector(frame, conf=self.conf_threshold)

            # Mesure du temps d'inférence
            inference_time = (time.time() - start_time) * 1000  # en ms

            # Parcours des détections
            plaques_detectees = 0
            for r in results:
                for box in r.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    plate_crop = frame[y1:y2, x1:x2]
                    if plate_crop.size == 0:
                        continue

                    plaques_detectees += 1

                    # Appliquer l'OCR seulement toutes les 'skip_frames' images
                    if frame_count % skip_frames == 0:
                        plate_text = self.recognizer.extract_text(plate_crop)
                        # Affichage dans la console du texte reconnu
                        print(f"Plaque détectée : {plate_text}")
                    else:
                        # Pour les frames intermédiaires, on pourrait réutiliser le dernier texte
                        # Mais ici on va juste afficher un message sans OCR pour ne pas ralentir
                        pass

                    # Dessiner sur l'image
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    if 'plate_text' in locals():
                        cv2.putText(frame, plate_text, (x1, y1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # Afficher le nombre de plaques détectées dans la console (similaire au format YOLO)
            if plaques_detectees == 0:
                print(f"{frame_count}: {frame.shape[1]}x{frame.shape[0]} (no detections), {inference_time:.1f}ms")
            else:
                print(
                    f"{frame_count}: {frame.shape[1]}x{frame.shape[0]} {plaques_detectees} license plate(s), {inference_time:.1f}ms")

            # Affichage vidéo
            cv2.imshow('License Plate Detection - Real Time', frame)
            out.write(frame)

            if cv2.waitKey(1) & 0xFF == ord('a'):
                break

        cap.release()
        out.release()
        cv2.destroyAllWindows()



if __name__ == '__main__':
    # Remplacez par le chemin de votre modèle spécialisé
    system = RealtimePlateSystem(model_path='license_plate_detector.pt', conf_threshold=0.5)
    system.run(source=1)  # 0 pour webcam, ou 'video.mp4'