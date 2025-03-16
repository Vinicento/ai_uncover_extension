import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights


import os
from torch.cuda.amp import GradScaler, autocast
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from sklearn.model_selection import train_test_split
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import time
import random


def initialize_model(num_classes=2):
        weights = EfficientNet_B0_Weights.DEFAULT
        model = efficientnet_b0(weights=weights)

        # Adding convolutional layers with smaller windows
        # Inserting batch normalization and ReLU activation after each convolutional layer
        model.features.add_module("conv_small_1", nn.Conv2d(model.features[-1].out_channels, 128, kernel_size=3, padding=1))
        model.features.add_module("bn_small_1", nn.BatchNorm2d(128))
        model.features.add_module("relu_small_1", nn.ReLU(inplace=True))

        model.features.add_module("conv_small_2", nn.Conv2d(128, 64, kernel_size=2, padding=1))
        model.features.add_module("bn_small_2", nn.BatchNorm2d(64))
        model.features.add_module("relu_small_2", nn.ReLU(inplace=True))

        model.features.add_module("conv_small_3", nn.Conv2d(64, 32, kernel_size=1, padding=1))
        model.features.add_module("bn_small_3", nn.BatchNorm2d(32))
        model.features.add_module("relu_small_3", nn.ReLU(inplace=True))

        model.features.add_module("conv_small_4", nn.Conv2d(32, 16, kernel_size=3, padding=1))
        model.features.add_module("bn_small_4", nn.BatchNorm2d(16))
        model.features.add_module("relu_small_4", nn.ReLU(inplace=True))

        model.features.add_module("conv_small_5", nn.Conv2d(16, 8, kernel_size=2, padding=1))
        model.features.add_module("bn_small_5", nn.BatchNorm2d(8))
        model.features.add_module("relu_small_5", nn.ReLU(inplace=True))

        model.features.add_module("conv_small_6", nn.Conv2d(8, 4, kernel_size=1, padding=1))
        model.features.add_module("bn_small_6", nn.BatchNorm2d(4))
        model.features.add_module("relu_small_6", nn.ReLU(inplace=True))

        # Adding a global average pooling layer to reduce spatial dimensions to 1x1
        model.features.add_module("global_avg_pool", nn.AdaptiveAvgPool2d(1))

        # Flatten the output for the classifier
        model.classifier[1] = nn.Sequential(
            nn.Flatten(),
            nn.Linear(4, num_classes),
            nn.Softmax(dim=1)  # Using Softmax for the final output
        )
        return model





def load_model(model_path, device):
    model = initialize_model(num_classes=2)
    state_dict = torch.load(model_path, map_location=device)

    # Handle the case where the model was saved with DataParallel
    if "module." in list(state_dict.keys())[0]:
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:]  # remove `module.`
            new_state_dict[name] = v
        state_dict = new_state_dict

    model.load_state_dict(state_dict)
    model = model.to(device)
    return model


def export_to_onnx(model, sample_input, export_path="model.onnx"):
    if isinstance(model, nn.DataParallel):
        model = model.module
    torch.onnx.export(model, sample_input, export_path, opset_version=9, input_names=['input'], output_names=['output'])


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = "ai_detection_model.pt"

    # Load the model
    model = load_model(model_path, device)

    # Prepare a sample input tensor
    sample_input = torch.randn(1, 3, 256, 256).to(device)  # Adjust dimensions if needed

    # Export the model to ONNX
    export_to_onnx(model, sample_input)


if __name__ == '__main__':
    main()
