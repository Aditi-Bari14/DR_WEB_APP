# inference/preprocess.py
from PIL import Image
import torch
from torchvision import transforms
import numpy as np

# Image transformations (same as training)
image_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

def preprocess(image_file, tabular_raw, device="cpu"):
    """
    Preprocess input for multimodal model.
    
    Parameters:
        image_file: file-like object (PIL image or uploaded file)
        tabular_raw: dictionary with keys: 'age', 'hba1c_latest', 'glucose_values'
    Returns:
        image_tensor, tabular_tensor
    """

    # --- 1. Preprocess image ---
    image = Image.open(image_file).convert("RGB")
    image_tensor = image_transform(image).unsqueeze(0).to(device)

    # --- 2. Preprocess tabular data ---
    # Extract raw inputs
    age = float(tabular_raw['age'])
    hba1c = float(tabular_raw['hba1c_latest'])
    glucose_values = [float(x.strip()) for x in tabular_raw['glucose_values'].split(",")]

    glucose_mean = np.mean(glucose_values)
    glucose_std = np.std(glucose_values)
    glucose_min = np.min(glucose_values)
    glucose_max = np.max(glucose_values)

    # Build tabular feature vector in correct order
    tabular_array = [age, hba1c, glucose_mean, glucose_std, glucose_min, glucose_max]

    tabular_tensor = torch.tensor([tabular_array], dtype=torch.float32).to(device)

    return image_tensor, tabular_tensor
