import torch
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F
from tqdm import tqdm
from robotControl import UR5_RG2

import threading

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.image_size = 227

        # AlexNet Pretained by ImageNet
        alexnet = models.alexnet(pretrained=True)
        conv_layers = list(alexnet.features.children())
        self.Conv_net = nn.ModuleList(conv_layers)
        self.flatten = nn.Flatten()
        linear_layers = list()
        linear_layers.append(nn.Linear(9216, 4096))
        linear_layers.append(nn.Linear(4096, 1024))
        self.Linear_net = nn.ModuleList(linear_layers)
        self.layerList = []
        # 最后的输出层
        for i in range(18):
            self.layerList.append(nn.Linear(1024, 2))
    
    def forward(self, input_pts):
        h = input_pts.clone().to(torch.float)
        # h = input_pts.clone().clone()
        for i, _ in enumerate(self.Conv_net):
            h = self.Conv_net[i](h)
        h = self.flatten(h)
        for i, _ in enumerate(self.Linear_net):
            h = F.relu(self.Linear_net[i](h))
        output = []
        for i in range(18):
            output.append(self.layerList[i](h).unsqueeze(1))
        output = torch.cat(output, 1)
        return output

class CustomLoss(nn.Module):
    '''自定义损失函数，只计算对应类别的损失'''
    def __init__(self):
        super(CustomLoss, self).__init__()

    def forward(self, outputs, labels, angle):
        # outputs是前面模型的输出，即36维的抓取可行性得分
        # labels是真实标签，表示真实的角度bin
        loss = 0
        output = outputs[torch.arange(outputs.shape[0]) ,angle, :]
        criterion = nn.CrossEntropyLoss()
        loss = criterion(output, labels)

        return loss

def importanceSampling():
    pass

# 初始化
robot = UR5_RG2()
# 训练参数
batch_size = 16
iterations = 1
epoch = 10

data, angle, target= robot.CollectSamples('Cuboid', batch_size)
angle = angle.long()
target = target.long()
net = Net()
for e in range(epoch):
    with tqdm(total=iterations, desc=f"Epoch {e}", ncols=100) as p_bar:
        # 前向传播
        trainResult = net(data)
        
        criterion = CustomLoss()
        loss = criterion(trainResult, target, angle)
        optimizer = torch.optim.Adam(net.parameters(), lr=0.01)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # 显示进度条
        p_bar.set_postfix({'loss': '{0:1.5f}'.format(loss.item())})
        p_bar.update(1)