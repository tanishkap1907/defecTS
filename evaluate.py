import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import timm
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, f1_score
from tqdm import tqdm

DATASET_PATH = "data"
IMAGE_SIZE = 224
BATCH_SIZE = 32
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

print(f"Using device: {DEVICE}")

# Same val transforms — no augmentation
val_transforms = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# Load dataset with same seed so val set is identical to training
full_dataset = datasets.ImageFolder(DATASET_PATH, transform=val_transforms)
class_names = full_dataset.classes
num_classes = len(class_names)

train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size
_, val_dataset = torch.utils.data.random_split(
    full_dataset, [train_size, val_size],
    generator=torch.Generator().manual_seed(42)
)

val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

# Load best model
model = timm.create_model('efficientnet_b0', pretrained=False, num_classes=num_classes, drop_rate=0.3)
model.load_state_dict(torch.load('best_model.pth', map_location=DEVICE))
model = model.to(DEVICE)
model.eval()
print("Model loaded successfully!")

# Get all predictions
all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels in tqdm(val_loader, desc="Evaluating"):
        images = images.to(DEVICE)
        outputs = model(images)
        _, predicted = outputs.max(1)
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.numpy())

all_preds = np.array(all_preds)
all_labels = np.array(all_labels)

# Overall accuracy
overall_acc = 100. * (all_preds == all_labels).sum() / len(all_labels)
print(f"\nOverall Accuracy: {overall_acc:.1f}%")

# Per class accuracy
print("\nPer Class Results:")
print("-" * 50)
per_class_acc = []
class_counts = []
for i, cls in enumerate(class_names):
    mask = all_labels == i
    if mask.sum() > 0:
        acc = 100. * (all_preds[mask] == all_labels[mask]).sum() / mask.sum()
        count = mask.sum()
        per_class_acc.append(acc)
        class_counts.append(count)
        print(f"{cls:15}: {acc:.1f}% accuracy ({count} images)")
    else:
        per_class_acc.append(0)
        class_counts.append(0)

# Print full classification report
print("\nFull Classification Report:")
print(classification_report(all_labels, all_preds, target_names=class_names))

# ── Plot 1: Confusion Matrix ──
plt.figure(figsize=(10, 8))
cm = confusion_matrix(all_labels, all_preds)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names, yticklabels=class_names)
plt.title('Confusion Matrix — defecTS Model', fontsize=14, fontweight='bold')
plt.ylabel('Actual Class')
plt.xlabel('Predicted Class')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('plot_confusion_matrix.png', dpi=150)
plt.show()
print("Saved: plot_confusion_matrix.png")

# ── Plot 2: Per Class Accuracy ──
colors = ['#E24B4A' if acc < 70 else '#EF9F27' if acc < 85 else '#1D9E75' for acc in per_class_acc]
plt.figure(figsize=(10, 5))
bars = plt.bar(class_names, per_class_acc, color=colors, edgecolor='white')
plt.axhline(y=85, color='navy', linestyle='--', linewidth=1.5, label='85% target')
plt.axhline(y=overall_acc, color='gray', linestyle='--', linewidth=1.5, label=f'Overall {overall_acc:.1f}%')
for bar, acc in zip(bars, per_class_acc):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
             f'{acc:.0f}%', ha='center', va='bottom', fontsize=9)
plt.title('Per Class Accuracy — defecTS Model', fontsize=14, fontweight='bold')
plt.xlabel('Defect Class')
plt.ylabel('Accuracy (%)')
plt.ylim(0, 110)
plt.xticks(rotation=45)
plt.legend()
plt.tight_layout()
plt.savefig('plot_per_class_accuracy.png', dpi=150)
plt.show()
print("Saved: plot_per_class_accuracy.png")

# ── Plot 3: Accuracy vs Class Occurrence ──
plt.figure(figsize=(8, 5))
scatter_colors = ['#E24B4A' if acc < 70 else '#EF9F27' if acc < 85 else '#1D9E75' for acc in per_class_acc]
plt.scatter(class_counts, per_class_acc, c=scatter_colors, s=120, zorder=5)
for i, cls in enumerate(class_names):
    plt.annotate(cls, (class_counts[i], per_class_acc[i]),
                textcoords="offset points", xytext=(8, 4), fontsize=8)
plt.axhline(y=85, color='navy', linestyle='--', linewidth=1.5, label='85% target')
plt.title('Accuracy vs Class Occurrence', fontsize=14, fontweight='bold')
plt.xlabel('Number of Images in Class')
plt.ylabel('Accuracy (%)')
plt.ylim(0, 110)
plt.legend()
plt.tight_layout()
plt.savefig('plot_accuracy_vs_occurrence.png', dpi=150)
plt.show()
print("Saved: plot_accuracy_vs_occurrence.png")

print("\nAll plots saved successfully!")
print(f"Final Overall Accuracy: {overall_acc:.1f}%")