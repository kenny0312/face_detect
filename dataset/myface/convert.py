import os
import numpy as np
import glob
from PIL import Image
from torchvision import transforms
import torch
import cv2
from facenet.model import FaceNetModel
facenet = FaceNetModel()

image_path = "image"
path = glob.glob(image_path + "/*.jpg")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def convert():
    if not len(path):
        print("没有图片")

    else:
        for i in path:
            image = cv2.imread(i)
            image = cv2.resize(image, (160,160)) / 127.5 - 1.0
            image = np.transpose(image, (2, 0, 1))
            tensor = torch.from_numpy(image).float()
            tensor = tensor.unsqueeze(0)

            filename = os.path.basename(i)  # 如 face1.jpg
            name = os.path.splitext(filename)[0]  # 如 face1

            with torch.no_grad():
                emb = facenet(tensor.to(device))
            torch.save(emb.cpu(), f"pth/{name}.pth")


if __name__ == "__main__":
    convert()