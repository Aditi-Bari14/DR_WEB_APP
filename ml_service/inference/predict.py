# inference/predict.py
# import torch
# from inference.load_models import load_model
# from inference.preprocess import preprocess

# # Load model once
# device = "cpu"
# model = load_model(device=device)

# def run_prediction(image_file, age, hba1c, glucose_values):
#     """
#     image_file: uploaded retina image
#     age: int or float
#     hba1c: float
#     glucose_values: string from form, e.g. "100,110,120"
#     """

#     # --- 1. Build tabular dict ---
#     tabular_raw = {
#         "age": age,
#         "hba1c_latest": hba1c,
#         "glucose_values": glucose_values
#     }

#     # --- 2. Preprocess ---
#     image_tensor, tabular_tensor = preprocess(image_file, tabular_raw, device=device)

#     # --- 3. Create dummy graph tensors for GCN ---
#     B = 1
#     gcn_features = torch.zeros((B, tabular_tensor.shape[1]), device=device)
#     edge_index = torch.zeros((2, 1), dtype=torch.long)  # dummy
#     node_idx = torch.zeros(B, dtype=torch.long, device=device)

#     # --- 4. Forward pass ---
#     with torch.no_grad():
#         output = model(image_tensor, tabular_tensor, gcn_features, edge_index, node_idx)
#         logits = output["logits"]
#         probs = torch.softmax(logits, dim=1)
#         confidence, pred_class = torch.max(probs, dim=1)

#     # --- 5. Map prediction ---
#     class_mapping = {0: "No DR", 1: "Mild", 2: "Moderate", 3: "Severe", 4: "Proliferative DR"}
#     prediction = class_mapping[int(pred_class)]
#     confidence = float(confidence)

#     return {"prediction": prediction, "confidence": confidence}
import torch
import cv2
import numpy as np
import os

from inference.load_models import load_model
from inference.preprocess import preprocess
from inference.lesion_explanation import generate_lesion_text

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

    predicted_class = int(pred_class.item())

    # -----------------------------
    # 3. Lesion-based explanation
    # -----------------------------
    lesion_report = generate_lesion_text(predicted_class)

    # 🔹 Save for Grad-CAM
    _last_image_tensor = image_tensor
    _last_predicted_class = predicted_class

    # -----------------------------
    # 4. Class mapping (IMPORTANT)
    # -----------------------------
    class_mapping = {
        0: "No DR",
        1: "Mild DR",
        2: "Moderate DR",
        3: "Severe DR",
        4: "Proliferative DR"
    }

    # -----------------------------
    # 5. Prepare response (FINAL FIX)
    # -----------------------------
    return {
        "prediction_class": predicted_class,              # ✅ INTEGER (0–4)
        "prediction": class_mapping[predicted_class],     # ✅ STRING
        "confidence": float(confidence.item()),
        "gradcam_image": None,        # generated later
        "prototype_image": None,      # generated later
        "lesion_report": lesion_report
    }


# -------------------- Grad-CAM Helper --------------------
def get_last_image_tensor_and_class():
    """
    Used by app.py for Grad-CAM generation
    """
    return _last_image_tensor, _last_predicted_class

