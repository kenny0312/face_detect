from torchvision.models.resnet import resnet50
from torch import nn
import torch
from torch.nn import functional as F


class FaceNetModel(nn.Module):
    def __init__(self, emd_size=256, class_nums=1000): #emd_size是输出的特征dimensions
        # 调用nn.module的父类
        super().__init__()

        self.emd_size = emd_size
        self.resnet = resnet50()

        self.faceNet = nn.Sequential(
            self.resnet.conv1,
            self.resnet.bn1,
            self.resnet.relu,
            self.resnet.maxpool,

            self.resnet.layer1,
            self.resnet.layer2,
            self.resnet.layer3,
            self.resnet.layer4,
            #拉平高维的4d特征图转换为2d向量喂给全连接层
            nn.Flatten(),
        )

        #设计全连接层，讲resnet输出的高维特征转换为emd_size
        self.fc = nn.Linear(32768, emd_size)

        self.l2_norm = F.normalize
        self.fc_class = nn.Linear(emd_size, class_nums)


    def forward(self, x):
        x = self.faceNet(x)
        # print(x.shape)
        # #用来的得到输入fully connected layer的 features

        x = self.fc(x)
        x = self.l2_norm(x)*10
        #print(x)
        return x

    #分类网络
    def forward_class(self,x):
        x = self.forward(x)
        x = self.fc_class(x)
        return x



if __name__ == '__main__':
    model = FaceNetModel()
    model(torch.zeros((2,3,128,128)))


