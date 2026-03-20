import torch
import timm

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

checkpoint = torch.load('best_model.pth', map_location=DEVICE)
print("Keys in checkpoint:", list(checkpoint.keys())[:5])
print("Total keys:", len(checkpoint.keys()))
print("Type:", type(checkpoint))