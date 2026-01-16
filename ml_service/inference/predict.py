import torch
import cv2
import numpy as np
import os

from inference.load_models import load_model
from inference.preprocess import preprocess

# -------------------- Setup --------------------
device = "cpu"
model = load_model(device=device)
model.eval()

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# 🔹 Globals for Grad-CAM access
_last_image_tensor = None
_last_predicted_class = None


# -------------------- Prediction --------------------
def run_prediction(image_path, age, hba1c, glucose_values):

    global _last_image_tensor, _last_predicted_class

    # -----------------------------
    # 1. Preprocess inputs
    # -----------------------------
    tabular_raw = {
        "age": age,
        "hba1c_latest": hba1c,
        "glucose_values": glucose_values
    }

    image_tensor, tabular_tensor = preprocess(
        image_path,
        tabular_raw,
        device=device
    )

    # Dummy GCN inputs (as per your model)
    B = 1
    gcn_features = torch.zeros((B, 6), device=device)
    edge_index = torch.zeros((2, 1), dtype=torch.long)
    node_idx = torch.zeros(B, dtype=torch.long, device=device)

    # -----------------------------
    # 2. FULL MODEL PREDICTION
    # -----------------------------
    with torch.no_grad():
        output = model(
            image_tensor,
            tabular_tensor,
            gcn_features,
            edge_index,
            node_idx
        )

    logits = output["logits"]
    probs = torch.softmax(logits, dim=1)
    confidence, pred_class = torch.max(probs, dim=1)

    # 🔹 Save for Grad-CAM (used later in ml_service)
    _last_image_tensor = image_tensor
    _last_predicted_class = pred_class.item()

    # -----------------------------
    # 3. Prepare response
    # -----------------------------
    class_mapping = {
        0: "No DR",
        1: "Mild",
        2: "Moderate",
        3: "Severe",
        4: "Proliferative DR"
    }

    return {
        "prediction": class_mapping[int(pred_class.item())],
        "confidence": float(confidence.item()),
        "gradcam_image": None,      # generated in ml_service
        "prototype_image": None
    }


# -------------------- Grad-CAM Helper --------------------
def get_last_image_tensor_and_class():
    """
    Used by ml_service/app.py for Grad-CAM generation
    """
    return _last_image_tensor, _last_predicted_class
