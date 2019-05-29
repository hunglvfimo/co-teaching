import math
import torch
import torch.nn as nn
import torch.nn.init as init 
import torch.nn.functional as F
import torch.optim as optim

from layers import *

class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=512, return_embedding=False):
        self.return_embedding = return_embedding
        
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        #self.maxpool = nn.MaxPool2d(kernel_size=3, stride=1, padding=1) # previous stride is 2
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(14)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                #m.weight.data.fill_(1)
                #m.bias.data.zero_()
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.get_embedding(x)
        if self.return_embedding:
            return x
        
        logit = self.fc(x)
        return logit
    
    def get_embedding(self, x):
        # x = self.bn0(x)
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        # x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        # x = self.flatten(x)
        x = x.view(x.size(0), -1)
        return x

class CoTeachingCNN(nn.Module):
    def __init__(self, input_channel=3, num_classes=10, dropout_rate=0.25, return_embedding=False):
        self.dropout_rate = dropout_rate
        self.return_embedding = return_embedding

        super(CoTeachingCNN, self).__init__()
        
        self.c1=nn.Conv2d(input_channel,128,kernel_size=3,stride=1, padding=1)
        self.c2=nn.Conv2d(128,128,kernel_size=3,stride=1, padding=1)
        self.c3=nn.Conv2d(128,128,kernel_size=3,stride=1, padding=1)
        self.c4=nn.Conv2d(128,256,kernel_size=3,stride=1, padding=1)
        self.c5=nn.Conv2d(256,256,kernel_size=3,stride=1, padding=1)
        self.c6=nn.Conv2d(256,256,kernel_size=3,stride=1, padding=1)
        self.c7=nn.Conv2d(256,512,kernel_size=3,stride=1, padding=0)
        self.c8=nn.Conv2d(512,256,kernel_size=3,stride=1, padding=0)
        self.c9=nn.Conv2d(256,128,kernel_size=3,stride=1, padding=0)

        self.dropout1=nn.Dropout2d(p=self.dropout_rate)
        self.dropout2=nn.Dropout2d(p=self.dropout_rate)
        
        self.flatten = Flatten()

        self.l_c1=nn.Linear(128, num_classes)
        
        self.bn1=nn.BatchNorm2d(128)
        self.bn2=nn.BatchNorm2d(128)
        self.bn3=nn.BatchNorm2d(128)
        self.bn4=nn.BatchNorm2d(256)
        self.bn5=nn.BatchNorm2d(256)
        self.bn6=nn.BatchNorm2d(256)
        self.bn7=nn.BatchNorm2d(512)
        self.bn8=nn.BatchNorm2d(256)
        self.bn9=nn.BatchNorm2d(128)

    def forward(self, x):
        x = self.get_embedding(x)
        if self.return_embedding: 
            return x
        
        logit = self.l_c1(x)
        return logit

    def get_embedding(self, x,):
        h=x
        h=self.c1(h)
        h=F.leaky_relu(self.bn1(h), negative_slope=0.01)
        h=self.c2(h)
        h=F.leaky_relu(self.bn2(h), negative_slope=0.01)
        h=self.c3(h)
        h=F.leaky_relu(self.bn3(h), negative_slope=0.01)
        h=F.max_pool2d(h, kernel_size=2, stride=2)
        h=self.dropout1(h)

        h=self.c4(h)
        h=F.leaky_relu(self.bn4(h), negative_slope=0.01)
        h=self.c5(h)
        h=F.leaky_relu(self.bn5(h), negative_slope=0.01)
        h=self.c6(h)
        h=F.leaky_relu(self.bn6(h), negative_slope=0.01)
        h=F.max_pool2d(h, kernel_size=2, stride=2)
        h=self.dropout2(h)

        h=self.c7(h)
        h=F.leaky_relu(self.bn7(h), negative_slope=0.01)
        h=self.c8(h)
        h=F.leaky_relu(self.bn8(h), negative_slope=0.01)
        h=self.c9(h)
        h=F.leaky_relu(self.bn9(h), negative_slope=0.01)
        
        h=F.avg_pool2d(h, kernel_size=h.data.shape[2])
        h = self.flatten(h)

        return h
