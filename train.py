import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import timm
import numpy as np
from tqdm import tqdm

DATASET_PATH = "data"
IMAGE_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 15
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

print(f"Using device: {DEVICE}")

train_transforms = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(30),
    transforms.ColorJitter(brightness=0.4, contrast=0.4),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

val_transforms = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

full_dataset = datasets.ImageFolder(DATASET_PATH, transform=train_transforms)
class_names = full_dataset.classes
num_classes = len(class_names)
print(f"Classes: {class_names}")
print(f"Total images: {len(full_dataset)}")

train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(
    full_dataset, [train_size, val_size],
    generator=torch.Generator().manual_seed(42)
)
val_dataset.dataset.transform = val_transforms

all_targets = [full_dataset.targets[i] for i in train_dataset.indices]
class_counts = np.bincount(all_targets, minlength=num_classes)
class_weights = 1.0 / np.maximum(class_counts, 1)
class_weights = class_weights / class_weights.sum() * num_classes

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

# Load checkpoint cleanly
model = timm.create_model('efficientnet_b0', pretrained=False, num_classes=num_classes, drop_rate=0.3)
model.load_state_dict(torch.load('best_model.pth', map_location=DEVICE))
model = model.to(DEVICE)
print("Loaded checkpoint successfully!")

# Unfreeze all
for param in model.parameters():
    param.requires_grad = True

class_weights_tensor = torch.tensor(class_weights, dtype=torch.float).to(DEVICE)

# Simple loss — no label smoothing this time
criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)

# Single learning rate for everything — simple and stable
optimizer = torch.optim.Adam(model.parameters(), lr=5e-5, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-7)

best_val_acc = 83.9

for epoch in range(EPOCHS):
    model.train()
    train_correct, train_total = 0, 0

    for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]"):
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        _, predicted = outputs.max(1)
        train_correct += predicted.eq(labels).sum().item()
        train_total += labels.size(0)

    model.eval()
    val_correct, val_total = 0, 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            _, predicted = outputs.max(1)
            val_correct += predicted.eq(labels).sum().item()
            val_total += labels.size(0)

    train_acc = 100. * train_correct / train_total
    val_acc = 100. * val_correct / val_total
    print(f"\nEpoch {epoch+1}: Train Acc={train_acc:.1f}% | Val Acc={val_acc:.1f}%")

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), 'best_model.pth')
        torch.save({'class_names': class_names}, 'class_info.pth')
        print(f"  Model saved! Best: {best_val_acc:.1f}%")

    scheduler.step()

print(f"\nTraining complete! Best validation accuracy: {best_val_acc:.1f}%")