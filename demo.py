import torch
from open3d.cpu.pybind.io import read_image
from torch import nn
import cv2
from PIL import Image
from model import FaceNetModel
from facenet_pytorch import MTCNN
import glob
import math
import torch.nn.functional as F
from config import Config
import numpy as np


config = Config.from_json_file("config.json")

detector = MTCNN(keep_all=True, device='cuda' if torch.cuda.is_available() else 'cpu')

def cv_to_pil(img):
    return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

def pil_to_cv(img):
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

#加载模型
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = FaceNetModel().to(device)
model.load_state_dict(torch.load("face_model.pth", map_location=device))
model.eval()


def read_image(path):
    image = cv2.imread(path)
    image = cv2.resize(image, (config.image_size,config.image_size)) / 127.5 - 1.0
    image = np.transpose(image, (2, 0, 1))
    return image

def face_net(img1, img2):
    #图片处理
    img1 = read_image(img1)
    img2 = read_image(img2)

    #人脸特征提取
    pred_arr1, pred_arr2 = model(img1), model(img2)
    d = F.pairwise_distance(pred_arr1, pred_arr2)
    return d


#人脸矫正，对齐
def face_cor(img):
    #bbox左上->右下 xyxy
    #landmark 眼鼻嘴 xxxxyyyy（右眼的x 左眼x 鼻子x 嘴巴x）
    try:
        bbox, landmarks = detector.detect(img)
    except:
        cv_to_pil(img)
        bbox, landmarks = detector.detect(img)

    left_eye = [landmarks[0][1], landmarks[0][6]]
    right_eye = [landmarks[0][0], landmarks[0][5]]
    k = (left_eye[1] - right_eye[1]) / (left_eye[0] - right_eye[0])
    arc = math.atan(k)
    angle = arc * 180 / math.pi
    #旋转
    img = img.rotate(angle)
    #
    face_img = img.crop((bbox[0][0], bbox[0][1], bbox[0][2], bbox[0][3]))
    return pil_to_cv(face_img)


def main():
    s_d = 0.3
    #开启摄像头
    for video_num in range(4):
        cap = cv2.VideoCapture() #也可以输入视频文件
        if cap.isOpened():
            print("开启了摄像头".format(video_num))

        else:
            assert"失败"

    #判断人脸库是否有样本
    if not len(glob.glob("./MyFace/*")):
        #人脸录入
        ret, img = cap.read()
        #人脸矫正
        old_img = face_cor(img)
        #保存人脸
        img = cv2.imwrite("./MyFace/face.jpg", old_img)

    else:
        old_face = cv2.imread("./MyFace/face.jpg")

    #人脸识别
    ret, new_face = cap.read()
    d = face_net(old_face, new_face)
    if d < s_d:
        print("通过")

    else:
        print("404")


if __name__ == '__main__':
    main()