import torch
import torch.nn as nn
from torchvision import models



# Creating ResNet50 model
class ResNet50(nn.Module):
    def __init__(self, num_classes, pretrained=True):
        super(ResNet50, self).__init__()
        self.resnet = models.resnet50(pretrained=pretrained)
        
        # Modify the final fully connected layer for your number of classes
        num_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_features, num_classes) 
        
    def forward(self, x):
        return self.resnet(x)
    
class TinyVGG(nn.Module):
    def __init__(self, input_shape: int, hidden_units: int, output_shape: int):
        super().__init__()
        self.conv_block_1 = nn.Sequential(
            nn.Conv2d(in_channels=input_shape, 
                      out_channels=hidden_units, 
                      kernel_size=3, 
                      stride=1, 
                      padding=0),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units, 
                      out_channels=hidden_units, 
                      kernel_size=3, 
                      stride=1,
                      padding=0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.conv_block_2 = nn.Sequential(
            nn.Conv2d(in_channels=hidden_units, 
                      out_channels=hidden_units, 
                      kernel_size=3, 
                      stride=1,
                      padding=0),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units, 
                      out_channels=hidden_units, 
                      kernel_size=3, 
                      stride=1,
                      padding=0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(hidden_units *53 * 53, output_shape)
        )
    
    def forward(self, x):
        x = self.conv_block_1(x)
        # print(f"conv_block_1: {x.shape}")
        x = self.conv_block_2(x)
        # print(f"conv_block_2: {x.shape}")
        x = self.classifier(x)
        # print(f"classifier: {x.shape}")
        return x