import cv2
import torch.nn.functional as F
from config import Config
import numpy as np
import torch
config = Config.from_json_file("../config.json")
from facenet_pytorch import MTCNN, extract_face
from PIL import Image, ImageDraw, ImageFont

mtcnn = MTCNN(keep_all=True)
img = cv2.imread("C:\\Users\\User\\PycharmProjects\\face_detect\\dataset\\100-0.JPG")

img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


# 转换为 PIL.Image（MTCNN 内部默认处理的是 PIL 图像）
img_pil = Image.fromarray(img_rgb)
boxes, probs, points = mtcnn.detect(img_pil, landmarks=True)
print(boxes, probs, points)


img_draw = img_pil.copy()
draw = ImageDraw.Draw(img_draw)
for i, (box, point) in enumerate(zip(boxes, points)):
    draw.rectangle(box.tolist(), width=5)
    for p in point:
        draw.rectangle((p - 10).tolist() + (p + 10).tolist(), width=10)
    extract_face(img, box, save_path='detected_face_{}.png'.format(i))


#
# from facenet.model import FaceNetModel
#
# model = FaceNetModel()

# def read_image(path):
#     image = cv2.imread(path)
#     image = cv2.resize(image, (config.image_size,config.image_size)) / 127.5 - 1.0
#     image = np.transpose(image, (2, 0, 1))
#     return image
#
#
# img1 = read_image("../dataset/CASIA_data/100-0.jpg")
# tensor1 = torch.from_numpy(img1).float()
# tensor1 = tensor1.unsqueeze(0)
#
#
#
# img2 = read_image("../dataset/CASIA_data/101-1.jpg")
# tensor2 = torch.from_numpy(img2).float()
# tensor2 = tensor2.unsqueeze(0)
#
# pred_arr1, pred_arr2 = model(tensor1), model(tensor2)
# d = F.pairwise_distance(pred_arr1, pred_arr2)
# print(d)
