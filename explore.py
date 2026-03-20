import os
import matplotlib.pyplot as plt
from PIL import Image

# Path to your dataset
DATASET_PATH = "data"

# Count images in each class
classes = sorted(os.listdir(DATASET_PATH))
classes = [c for c in classes if not c.startswith('.')]  # remove hidden files

print("Classes found:", classes)
print("\nImage count per class:")
print("-" * 30)

counts = {}
for cls in classes:
    cls_path = os.path.join(DATASET_PATH, cls)
    images = [f for f in os.listdir(cls_path) if not f.startswith('.')]
    counts[cls] = len(images)
    print(f"{cls:15} : {len(images)} images")

print("-" * 30)
print(f"Total images: {sum(counts.values())}")

# Plot class distribution
plt.figure(figsize=(10, 5))
plt.bar(counts.keys(), counts.values(), color='steelblue')
plt.title('Image Count per Defect Class')
plt.xlabel('Class')
plt.ylabel('Number of Images')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('class_distribution.png')
plt.show()
print("\nChart saved as class_distribution.png")

# Show one sample image from each class
fig, axes = plt.subplots(1, len(classes), figsize=(20, 4))
for i, cls in enumerate(classes):
    cls_path = os.path.join(DATASET_PATH, cls)
    images = [f for f in os.listdir(cls_path) if not f.startswith('.')]
    img = Image.open(os.path.join(cls_path, images[0]))
    axes[i].imshow(img, cmap='gray')
    axes[i].set_title(cls, fontsize=9)
    axes[i].axis('off')
plt.tight_layout()
plt.savefig('sample_images.png')
plt.show()
print("Sample images saved as sample_images.png")