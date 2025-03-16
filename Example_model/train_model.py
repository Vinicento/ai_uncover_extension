import torch
import torch.nn as nn
import torch.optim as optim

# Define a simple CNN model for binary classification
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(32*8*8, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.flatten(x)
        x = self.fc(x)
        x = self.sigmoid(x)  # Output probability of being manipulated
        return x

# Generate random data
inputs = torch.randn(1, 3, 32, 32)  # Batch size 1, 3 channels, 32x32 image size

# Instantiate the model
model = SimpleCNN()

# Define loss function and optimizer
criterion = nn.BCELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Example training loop
for epoch in range(10):
    optimizer.zero_grad()
    outputs = model(inputs)
    labels = torch.rand(1, 1)  # Random binary labels
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()

# Export the model to ONNX format
torch.onnx.export(model, inputs, "model.onnx", opset_version=9)
