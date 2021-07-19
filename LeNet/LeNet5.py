import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self, n_classes):
        super(Net, self).__init__()

        self.feature_extractor = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=6,kernel_size=5,stride=1),#Convolution
            nn.Tanh(), #Activation
            nn.AvgPool2d(kernel_size=2), #Sub Sampling

            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1), #Convolution
            nn.Tanh(), #Activation
            nn.AvgPool2d(kernel_size=2), #Sub Sampling

            nn.Conv2d(in_channels=16, out_channels=120, kernel_size=5, stride=1),  # Convolution
            nn.Tanh(),  # Activation
        )

        self.classifier = nn.Sequential(
            nn.Linear(in_features=120, out_features=84), #Fully Connected
            nn.Tanh(), # Activation
            nn.Linear(in_features=84, out_features=n_classes) #Fully Connected
        )

    def forward(self, x):

        x = self.feature_extractor(x)
        x = torch.flatten(x, 1) #다차원 배열을 1차원 배열로 펴주는 작업 (View 함수와 동일)
        logits = self.classifier(x)
        probs = F.softmax(logits, dim=1)
        return logits, probs