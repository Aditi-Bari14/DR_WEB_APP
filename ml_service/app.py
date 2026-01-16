from flask import Flask, request, jsonify, send_from_directory
import os
import json
from datetime import datetime
import torch
import cv2
import numpy as np

from inference.predict import run_prediction, get_last_image_tensor_and_class
from inference.gradcam import GradCAM
from inference.load_models import load_model

# -------------------- Flask app setup --------------------
app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
STATIC_FOLDER = "static"
GRADCAM_FOLDER = os.path.join(STATIC_FOLDER, "gradcam")

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(GRADCAM_FOLDER, exist_ok=True)

HISTORY_FILE = "history.json"

# -------------------- Load Model ONCE --------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Loading multimodal DR model...")
model = load_model(device=device)
model.eval()

# 🔴 IMPORTANT for GradCAM
def disable_inplace_relu(module):
    if isinstance(module, torch.nn.ReLU):
        module.inplace = False

model.apply(disable_inplace_relu)

# -------------------- Grad-CAM Setup --------------------
cnn_backbone = model.cnn
cnn_backbone.eval()

target_layer = model.cnn.features.denseblock4
gradcam = GradCAM(cnn_backbone, target_layer)

# -------------------- History Helper --------------------
def save_history(record):
    if not os.path.exists(HISTORY_FILE) or os.stat(HISTORY_FILE).st_size == 0:
        history = []
    else:
        try:
            with open(HISTORY_FILE, "r") as f:
                history = json.load(f)
        except json.JSONDecodeError:
            history = []

    history.append(record)

    with open(HISTORY_FILE, "w") as f:
        json.dump(history, f, indent=4)

# -------------------- Prediction Route --------------------
@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    image = request.files["image"]
    image_path = os.path.join(UPLOAD_FOLDER, image.filename)
    image.save(image_path)

    try:
        age = float(request.form.get("age"))
        hba1c = float(request.form.get("hba1c"))
        glucose_values = request.form.get("glucose_values")
    except Exception as e:
        return jsonify({"error": f"Invalid input: {e}"}), 400

    try:
        result = run_prediction(image_path, age, hba1c, glucose_values)
        image_tensor, predicted_class = get_last_image_tensor_and_class()
    except Exception as e:
        return jsonify({"error": f"Prediction failed: {e}"}), 500

    # -------------------- Generate Patient ID (ONCE) --------------------
    if not os.path.exists(HISTORY_FILE) or os.stat(HISTORY_FILE).st_size == 0:
        patient_index = 1
    else:
        try:
            with open(HISTORY_FILE, "r") as f:
                history = json.load(f)
            patient_index = len(history) + 1
        except:
            patient_index = 1

    patient_id = f"PAT_{patient_index:04d}"

    # -------------------- Grad-CAM --------------------
    with torch.enable_grad():
        cam = gradcam.generate(image_tensor, predicted_class)

    gradcam_filename = f"gradcam_{patient_id}.jpg"
    gradcam_path = os.path.join(GRADCAM_FOLDER, gradcam_filename)

    img = cv2.imread(image_path)
    img = cv2.resize(img, (224, 224))

    heatmap = cv2.applyColorMap(
        np.uint8(255 * cam),
        cv2.COLORMAP_JET
    )

    overlay = cv2.addWeighted(img, 0.6, heatmap, 0.4, 0)

    cv2.imwrite(gradcam_path, overlay)
    print("✅ GradCAM saved:", gradcam_path)

    # -------------------- Save History --------------------
    save_history({
        "patient_id": patient_id,
        "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "image_name": image.filename,
        "gradcam_image": f"/static/gradcam/{gradcam_filename}",
        "prototype_image": None,
        "age": age,
        "hba1c": hba1c,
        "prediction": result["prediction"],
        "confidence": round(result["confidence"] * 100, 2)
    })

    result["patient_id"] = patient_id
    result["gradcam_image"] = f"/static/gradcam/{gradcam_filename}"

    return jsonify(result)

# -------------------- History Route --------------------
@app.route("/history", methods=["GET"])
def get_history():
    try:
        with open(HISTORY_FILE, "r") as f:
            history = json.load(f)
    except FileNotFoundError:
        history = []

    history = sorted(history, key=lambda x: x["date"])
    return jsonify(history)

# -------------------- Serve Uploads --------------------
@app.route("/uploads/<filename>")
def uploaded_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

# -------------------- Main --------------------
if __name__ == "__main__":
    print("Starting ML service...")
    app.run(port=8000, debug=True)
