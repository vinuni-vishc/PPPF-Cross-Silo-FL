import timm
import torch
import torch.nn.functional as F
from torch import nn
from models.resnet import ResNet18, ResNet34
from models.mobilenetv2 import MobileNetV2
from models.preact_resnet import PreActResNet18


class Model(nn.Module):
    def __init__(self, num_classes, image_size):
        super(Model, self).__init__()
        self.input_shape = (1, 3 * image_size * image_size)
        self.num_classes = num_classes
        self.backbone = timm.create_model('resnet18', pretrained=False, num_classes=self.num_classes)

    def forward(self, x):
        return self.backbone(x)
