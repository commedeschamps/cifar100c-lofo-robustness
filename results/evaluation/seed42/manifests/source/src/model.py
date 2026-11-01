"""Identical scratch architecture for all future experimental conditions."""

from torch import nn
from torchvision.models import resnet18


def build_model():
    model = resnet18(weights=None, num_classes=100)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    # Match torchvision's Kaiming initialization for the replacement convolution.
    nn.init.kaiming_normal_(model.conv1.weight, mode="fan_out", nonlinearity="relu")
    model.maxpool = nn.Identity()
    return model
