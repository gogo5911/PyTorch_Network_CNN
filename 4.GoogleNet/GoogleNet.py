import torch
import torch.nn as nn
import torch.nn.functional as F

class Auxiliary(nn.Module):
    def __init__(self, input_channels, n_classes):
        super(Auxiliary, self).__init__()

        self.Conv2 = nn.Conv2d(input_channels, 128, kernel_size=1)
        self.FC1 = nn.Linear(2048, 1024)
        self.FC2 = nn.Linear(1024, n_classes)

    def forward(self, x):
        # aux1: N x 512 x 14 x 14, aux2: N x 528 x 14 x 14
        x = F.adaptive_avg_pool2d(x, (4, 4))
        # aux1: N x 512 x 4 x 4, aux2: N x 528 x 4 x 4
        x = self.Conv2(x)
        # N x 128 x 4 x 4
        x = x.view(x.size(0), -1)
        # N x 2048
        x = F.relu(self.FC1(x), inplace=True)
        # N x 2048
        x = F.dropout(x, 0.7, training=self.training)
        # N x 2048
        x = self.FC2(x)
        # N x 1024
        return x


class Inception(nn.Module):
    def __init__(self, input_channels, n1x1, n3x3_reduce, n3x3, n5x5_reduce, n5x5, pool_proj):
        super(Inception, self).__init__()


        # 1x1conv branch
        self.inception_1 = nn.Sequential(
            nn.Conv2d(input_channels, n1x1, kernel_size=1)
        )

        self.inception_2 = nn.Sequential(
            nn.Conv2d(input_channels, n3x3_reduce, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(n3x3_reduce, n3x3, kernel_size=1),
            nn.ReLU(inplace=True)
        )

        self.inception_3 = nn.Sequential(
            nn.Conv2d(input_channels, n5x5_reduce, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(n5x5_reduce, n5x5, kernel_size=1),
            nn.ReLU(inplace=True)
        )

        self.inception_4 = nn.Sequential(
            nn.MaxPool2d(3, stride=1, padding=1),
            nn.Conv2d(input_channels, pool_proj, kernel_size=1),
            nn.ReLU(inplace=True)
        )


    def forward(self, x):
        return torch.cat((self.inception_1(x),self.inception_2(x),self.inception_3(x),self.inception_4(x)), dim=1)




class Net(nn.Module):
    def __init__(self, n_classes):
        super(Net, self).__init__()

        self.pre_layer = nn.Sequential(
            # N x 3 x 224 x 224
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            # N x 64 x 112 x 112
            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
            # N x 64 x 56 x 56
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            # N x 64 x 56 x 56
            nn.Conv2d(in_channels=64, out_channels=192, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True),
            # N x 192 x 56 x 56
            nn.MaxPool2d(kernel_size=3, stride=2, padding=0, ceil_mode=True),
        )

        # N x 192 x 28 x 28
        self.inception_3a = Inception(192, 64, 96, 128, 16, 32, 32)
        # N x 256 x 28 x 28
        self.inception_3b = Inception(256, 128, 128, 192, 32, 96, 64)
        # N x 480 x 28 x 28
        self.maxPool3 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # N x 480 x 14 x 14
        self.inception_4a = Inception(480, 192, 96, 208, 16, 48, 64)
        self.aux1 = Auxiliary(512, n_classes)
        # N x 512 x 14 x 14
        self.inception_4b = Inception(512, 160, 112, 224, 24, 64, 64)
        # N x 512 x 14 x 14
        self.inception_4c = Inception(512, 128, 128, 256, 24, 64, 64)
        # N x 512 x 14 x 14
        self.inception_4d = Inception(512, 112, 144, 288, 32, 64, 64)
        self.aux2 = Auxiliary(528, n_classes)
        # N x 528 x 14 x 14
        self.inception_4e = Inception(528, 256, 160, 320, 32, 128, 128)
        # N x 832 x 14 x 14
        self.maxPool4 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # N x 832 x 7 x 7
        self.inception_5a = Inception(832, 256, 160, 320, 32, 128, 128)
        # N x 832 x 7 x 7
        self.inception_5b = Inception(832, 384, 192, 384, 48, 128, 128)
        # N x 1024 x 7 x 7
        self.avgPool5 = nn.AvgPool2d(kernel_size=7, stride=1)

        # N x 1024 x 1 x 1
        self.dropout = nn.Dropout(p=0.4)
        # N x 1024
        self.linear = nn.Linear(in_features=1024, out_features=n_classes)



    def forward(self, x, mode=True):
        x = self.pre_layer(x)

        x = self.inception_3a(x)
        x = self.inception_3b(x)
        x = self.maxPool3(x)

        x = self.inception_4a(x)

        if mode:
            aux1 = self.aux1(x)
        else:
            aux1 = None

        x = self.inception_4b(x)
        x = self.inception_4c(x)
        x = self.inception_4d(x)

        if mode:
            aux2 = self.aux2(x)
        else:
            aux2 = None

        x = self.inception_4e(x)
        x = self.maxPool4(x)

        x = self.inception_5a(x)
        x = self.inception_5b(x)

        x = self.avgPool5(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.linear(x)
        x = F.softmax(x, dim=1)

        return x, aux1, aux2


