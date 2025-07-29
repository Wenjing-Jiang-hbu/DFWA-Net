from ultralytics import YOLO


model = YOLO("ultralytics/cfg/models/v8/yolov8s.yaml")

# Train the model
results = model.train(data="yourdataset", epochs=1000)
