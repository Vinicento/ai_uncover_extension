# Imports and setup
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
from torchvision.models import efficientnet_b1, EfficientNet_B1_Weights
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import time
import random


def initialize_model(device, num_classes=2):
    weights = EfficientNet_B1_Weights.DEFAULT
    model = efficientnet_b1(weights=weights)
    num_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_features, num_classes)
    model = model.to(device)
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs!")
        model = nn.DataParallel(model)
    return model


class Alaska2Dataset(Dataset):
    def __init__(self, filenames, labels, transform=None):
        self.filenames = filenames
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        image_path = self.filenames[idx]
        image = Image.open(image_path)
        image = image.convert('RGB')  # Convert to RGB only if necessary
        if self.transform:
            image = self.transform(image)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return image, label


def prepare_loaders(base_dir, transform, val_size=0.2, test_size=0.2, random_state=42, batch_size=32, num_workers=4):
    all_paths, all_labels = get_image_paths_and_labels(base_dir)

    train_paths, temp_paths, train_labels, temp_labels = train_test_split(
        all_paths, all_labels, test_size=val_size + test_size, random_state=random_state
    )

    valid_size = val_size / (val_size + test_size)
    valid_paths, test_paths, valid_labels, test_labels = train_test_split(
        temp_paths, temp_labels, test_size=1 - valid_size, random_state=random_state
    )

    train_dataset = Alaska2Dataset(train_paths, train_labels, transform)
    valid_dataset = Alaska2Dataset(valid_paths, valid_labels, transform)
    test_dataset = Alaska2Dataset(test_paths, test_labels, transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                              pin_memory=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                              pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                             pin_memory=True)

    return train_loader, valid_loader, test_loader


def get_image_paths_and_labels(base_dir):
    ai_images_dir = 'ai_images/ai_images'
    natural_images_dir = 'natural_images/natural_images'
    extensions = ('.jpeg', '.png', '.jpg')

    ai_dir_path = os.path.join(base_dir, ai_images_dir)
    natural_dir_path = os.path.join(base_dir, natural_images_dir)

    ai_paths = [os.path.join(ai_dir_path, file_name) for file_name in os.listdir(ai_dir_path) if
                file_name.endswith(extensions)]
    natural_paths = [os.path.join(natural_dir_path, file_name) for file_name in os.listdir(natural_dir_path) if
                     file_name.endswith(extensions)]

    # Balance dataset by oversampling
    if len(ai_paths) < len(natural_paths):
        ai_paths = random.choices(ai_paths, k=len(natural_paths))
    elif len(ai_paths) > len(natural_paths):
        natural_paths = random.choices(natural_paths, k=len(ai_paths))

    paths = ai_paths + natural_paths
    labels = [1] * len(ai_paths) + [0] * len(natural_paths)

    print(f"Found {len(paths)} images: {len(ai_paths)} AI and {len(natural_paths)} Natural")

    return paths, labels


def train_model(model, train_loader, valid_loader, criterion, optimizer, device, num_epochs=10, patience=10,
                save_path='ai_detection_model.pt'):
    scaler = GradScaler()
    best_loss = np.inf
    patience_counter = 0
    start_time = time.time()
    max_duration = 11 * 3600

    for epoch in range(num_epochs):
        if time.time() - start_time > max_duration:
            print("Training stopped due to time limit.")
            break

        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            with autocast():
                outputs = model(inputs)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item() * inputs.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)
        print(f'Epoch {epoch + 1}/{num_epochs}, Training Loss: {epoch_loss:.4f}')

        valid_loss = validate_model(model, valid_loader, criterion, device)
        print(f'Epoch {epoch + 1}/{num_epochs}, Validation Loss: {valid_loss:.4f}')

        if valid_loss < best_loss:
            best_loss = valid_loss
            torch.save(model.state_dict(), save_path)
            print("Saving the best model...")
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f'Early stopping! No improvement in validation loss for {patience} epochs.')
                break

    print("Training completed.")


def save_confusion_matrix_plot(true_labels, pred_labels, class_names, file_path):
    cm = confusion_matrix(true_labels, pred_labels)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.savefig(file_path)
    plt.close()


def validate_model(model, valid_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    total_samples = 0

    with torch.no_grad():
        for inputs, labels in valid_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)
            total_samples += inputs.size(0)

    valid_loss = running_loss / total_samples
    return valid_loss


def predict_probability(model, dataloader, device):
    model.eval()
    probabilities = []

    with torch.no_grad():
        for inputs, _ in dataloader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            probs = nn.Softmax(dim=1)(outputs)
            probabilities.extend(probs[:, 1].cpu().numpy())

    return probabilities


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    base_dir = '/kaggle/input/ai-images-detection'
    batch_size = 80
    num_workers = 4

    train_loader, valid_loader, test_loader = prepare_loaders(base_dir, transform, batch_size=batch_size,
                                                              num_workers=num_workers)

    model = initialize_model(device, num_classes=2)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    train_model(model, train_loader, valid_loader, criterion, optimizer, device, num_epochs=100, patience=15)

    true_labels, pred_labels = [], []
    with torch.no_grad():
        for inputs, labels in valid_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            true_labels.extend(labels.cpu().numpy())
            pred_labels.extend(preds.cpu().numpy())

    class_names = ['Natural', 'Ai']

    save_confusion_matrix_plot(true_labels, pred_labels, class_names, 'confusion_matrix_test.png')


if __name__ == '__main__':
    main()