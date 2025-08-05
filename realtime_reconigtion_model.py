from ultralytics import YOLO
import torch
from facenet.model import FaceNetModel
import glob
from config import Config
import cv2
from PIL import Image
from torchvision import transforms

import numpy as np

config = Config().from_json_file("config.json")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#初始化模型
#facenet
facenet = FaceNetModel(config.emd_size, config.class_nums).to(device)
facenet.load_state_dict(torch.load("facenet/facenet.pth2", map_location=device))
facenet.eval()

#yolo
yolo = YOLO('yolov8n.pt').to(device)
yolo.eval()


#提取myface的pth文件 以dict形式
myface = "dataset/myface/pth"
face_db = {}

path = glob.glob(myface + "/*.pth")

if not len(path):
    print("no face detected")
else:
    for p in path:
        name = p.rsplit(".pth", 1)[0]

        #加载tensor
        emb = torch.load(p)
        face_db[name] = emb

transform = transforms.Compose([
    transforms.Resize((160, 160)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

#打开摄像头/mp4文件
cap = cv2.VideoCapture("dataset/myface/video/masami.mp4")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("视频播放完毕或读取失败")
        break

    results = yolo.predict(frame,imgsz=640, conf=0.5, device=0 if torch.cuda.is_available() else "cpu", verbose=False)

    for result in results:
        boxes = result.boxes.xyxy.cpu().numpy()#左上右下坐标
        frame = result.orig_img#原始图片大小用来裁剪

        for box in boxes:
            #将坐标转成int
            x1, y1, x2, y2 = map(int, box)
            #在原始图片裁剪出框
            face_crop = frame[y1:y2, x1:x2]

            #太小的框不要
            if face_crop.shape[0] <30 or face_crop.shape[1]<30:
                continue

            #facenet to get emb features
            face_pil = Image.fromarray(cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB))
            face_tensor = transform(face_pil).unsqueeze(0).to(device)

            with torch.no_grad():#在使用facenet得到embedding过程中不计算梯度
                embeddings = facenet(face_tensor)


            #compare
            best_name = "Unknown"
            best_dist = float('inf')

            for name, db_emb in face_db.items():#访问字典 for key, values in dict
                dist = torch.norm(embeddings - db_emb).item()
                if dist < best_dist:
                    best_dist = dist
                    best_name = name

            label = best_name if best_dist < 20 else "Unknown"

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{label} ({best_dist:.2f})", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)


    cv2.imshow("Face Recognition", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()











