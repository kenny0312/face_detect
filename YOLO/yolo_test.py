from ultralytics import YOLO

model = YOLO('yolo11n.pt')  # 普通模型，不专门做人脸
model.predict("C:\\Users\\User\\Videos\\Captures\\Facebook - Google Chrome 2025-08-04 11-30-53.mp4", show=True)
