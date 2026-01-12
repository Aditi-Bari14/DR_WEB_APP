# inference/predict.py
import torch
from inference.load_models import load_model
from inference.preprocess import preprocess

# Load model once
device = "cpu"
model = load_model(device=device)

def run_prediction(image_file, age, hba1c, glucose_values):
    """
    image_file: uploaded retina image
    age: int or float
    hba1c: float
    glucose_values: string from form, e.g. "100,110,120"
    """

    # --- 1. Build tabular dict ---
    tabular_raw = {
        "age": age,
        "hba1c_latest": hba1c,
        "glucose_values": glucose_values
    }

    # --- 2. Preprocess ---
    image_tensor, tabular_tensor = preprocess(image_file, tabular_raw, device=device)

    # --- 3. Create dummy graph tensors for GCN ---
    B = 1
    gcn_features = torch.zeros((B, tabular_tensor.shape[1]), device=device)
    edge_index = torch.zeros((2, 1), dtype=torch.long)  # dummy
    node_idx = torch.zeros(B, dtype=torch.long, device=device)

    # --- 4. Forward pass ---
    with torch.no_grad():
        output = model(image_tensor, tabular_tensor, gcn_features, edge_index, node_idx)
        logits = output["logits"]
        probs = torch.softmax(logits, dim=1)
        confidence, pred_class = torch.max(probs, dim=1)

    # --- 5. Map prediction ---
    class_mapping = {0: "No DR", 1: "Mild", 2: "Moderate", 3: "Severe", 4: "Proliferative DR"}
    prediction = class_mapping[int(pred_class)]
    confidence = float(confidence)

    return {"prediction": prediction, "confidence": confidence}
