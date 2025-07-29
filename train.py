import torch
from sympy.abc import alpha
from torch import nn
from torch.utils import data as Data
import cv2
from glob import glob
from tqdm import tqdm
import re
from config import Config
import random
import numpy as np
from torch.nn import functional as F
from model import FaceNetModel



config = Config.from_json_file("config.json")

#构建自定义数据集（处理数据），配合data loader后续使用
class FaceData(Data.Dataset):
    def __init__(self,paths):
        self.paths = paths

    def to_tuple(self, x):
        return x,x

    def read_image(self,path):
        image = cv2.imread(path)
        image = cv2.resize(image, self.to_tuple(config.image_size)) / 127.5 - 1.0
        image = np.transpose(image, (2, 0, 1))
        return image

    def __getitem__(self, index):
        #a为anchor图 p为正类 n为负类
        a_path, p_path, n_path = self.paths[index]

        a_img, p_img, n_img = self.read_image(a_path), self.read_image(p_path), self.read_image(n_path)

        #提取label
        s_l= int(re.findall("(\d+)-\d+\.",a_path)[0]) -100
        n_l = int(re.findall("(\d+)-\d+\.",n_path)[0]) -100
        print("s_l 和n_L",s_l, n_l)

        return np.float32(a_img), np.float32(p_img), np.float32(n_img), np.int64(s_l), np.int64(n_l)

    def __len__(self):
        return len(self.paths)

class TripletLoss(nn.Module):
    def __init__(self,alpha):
        super().__init__()
        self.alpha = alpha
        self.pairwise_distance = nn.PairwiseDistance()

    def forward(self,a_x,p_x, n_x):
        s_d = self.pairwise_distance(a_x, p_x)
        n_d = self.pairwise_distance(a_x, n_x)
        return torch.clamp(s_d - n_d + self.alpha, min = 0.0)


#数据读取
def read_data():
    paths = glob("./dataset/*")

    #数据重构
    #人脸图片路径归类
    paths_dict = dict()

    for path in paths:
        man_num = re.findall("(\d+)-\d+\.",path)[0]
        if man_num not in paths_dict:
            paths_dict[man_num] = [path]

        else:
            paths_dict[man_num].append(path)


    #构建数据
    new_paths = []
    #构建一个list，前两张是同一个人，后一张不同人脸
    keys = list(paths_dict.keys())
    config.class_nums = len(keys)
    for i in range(config.class_nums):
        paths_ls = paths_dict[keys[i]]

        paths_lens = len(paths_ls)
        for image_path in paths_ls:
            new_path = [image_path]
            #同类
            rand_num = random.randint(0,paths_lens-1)
            new_path.append(paths_ls[rand_num])

            #负类
            rand_num_newman =random.randint(0, config.class_nums-1)
            if rand_num_newman == i :
                try:
                    rand_num_newman +=1
                    n_keys = keys[rand_num_newman]

                except:
                    rand_num_newman -= 1
                    n_keys = keys[rand_num_newman]
            else:
                n_keys = keys[rand_num_newman]

            n_path = paths_dict[n_keys]
            #从负类中再重新选择一个样本
            n_num =random.randint(0, len(n_path)-1)
            new_path.append(n_path[n_num])

            new_paths.append(new_path)
    print(new_paths)
    return new_paths

def train():
    paths = read_data()
    train_data = FaceData(paths)
    train_data_new = Data.DataLoader(dataset=train_data, batch_size=config.batch_size, shuffle=True)

    #初始化网络
    model = FaceNetModel(config.emd_size, config.class_nums)
    model.train()#设置模式为训练模式

    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    loss_t = TripletLoss(config.alpha)
    loss_c = nn.CrossEntropyLoss()

    #训练
    nb = len(train_data)
    for epoch in range(1, config.epochs+1):
        #可视化
        pbar = tqdm(train_data_new, total =nb)
        #
        for step, (a_x, p_x, n_x, s_y, n_y) in enumerate(pbar):
            a_out, p_out, n_out = model(a_x), model(p_x), model(n_x)

            s_d = F.pairwise_distance(a_out, p_out)
            n_d = F.pairwise_distance(a_out, n_out)

            thing = (n_d - s_d < config.alpha).flatten()
            mask = thing.nonzero(as_tuple=True)[0]

            if not len(mask):
                continue
            #计算三元组损失
            a_out,p_out,n_out = a_out[mask],p_out[mask],n_out[mask]
            loss_t_value = torch.mean(loss_t(a_out,p_out,n_out))

            #计算熵损失
            a_x, p_x, n_x = a_x[mask], p_x[mask], n_x[mask]
            input_x = torch.cat([a_x, p_x, n_x])
            s_y, n_y = s_y[mask], n_y[mask]
            output_y = torch.cat([s_y,s_y,n_y])

            out = model.forward_class(input_x)
            loss_c_value = loss_c(out,output_y)

            loss= loss_c_value + loss_t_value

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            s = ("train ===> epoch:{} ---- step:{} ---- loss_t_value:{:.4f} ----loss_c:{:.4f} ----loss:{:.4f}".format(epoch,step,loss_t_value.item(),loss_c_value.item(),loss.item()))

            pbar.set_description(s)

        #保存
        if not epoch % 2 and epoch>2:
            torch.save(model.state_dict(), "facenet.pth")

if __name__ == '__main__':
    train()

