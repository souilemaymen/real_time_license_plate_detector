from ultralytics import YOLO

def train_model():
    model = YOLO("yolov8n.pt")
    results = model.train(data='dataset/data.yaml',epoch =50, imgsz = 640,batch=16,name='plate_detector')