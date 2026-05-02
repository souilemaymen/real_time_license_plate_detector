import cv2
from ultralytics import YOLO
from recognizer import PlateRecognizer

class LicensePlateSystem :
    def __init__(self,model_path):
        self.detector = YOLO(model_path)
        self.recognizer = PlateRecognizer(languages=["en"])
        self.conf_threshold = 0.5
    def process_image(self,image_path):
        img = cv2.imread(image_path)
        detections = self.detector(img,conf=self.conf_threshold)
        for r in detections:
            for box in r.boxes:
                x1,y1,x2,y2 = map(int,box.xyxy[0])
                plate_img = img[y1:y2 , x1,x2]
                if plate_img.size != 0:
                    plate_text = self.recognizer.extract_text(plate_img)
                    print(f"plaque detecté: {plate_text}")
                    cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)
                    cv2.putText(img, plate_text, (x1, y1-10),cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)

        cv2.imwrite('results.jpg',img)
        return img



