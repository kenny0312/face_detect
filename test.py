from ultralytics import YOLO

model = YOLO("yolov8n.pt")
results = model.predict("C:\\Users\\User\\Videos\\Captures\\Facebook - Google Chrome 2025-08-04 11-30-53.mp4")

# 获取原图尺寸（H, W, C）
orig_shape = results[0].orig_img.shape
print(f"原始尺寸: {orig_shape}")