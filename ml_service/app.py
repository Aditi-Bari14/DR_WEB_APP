from flask import Flask, request, jsonify, send_from_directory
import os
import json
from datetime import datetime
import torch
import cv2
import numpy as np
import shutil

from inference.predict import run_prediction, get_last_image_tensor_and_class
from inference.gradcam import GradCAM
from inference.load_models import load_model
from inference.prototype_similarity import find_most_similar_prototype
from inference.lesion_explanation import generate_lesion_text


# -------------------- Flask app setup --------------------
app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
STATIC_FOLDER = "static"
GRADCAM_FOLDER = os.path.join(STATIC_FOLDER, "gradcam")
PROTOTYPE_OUTPUT_FOLDER = os.path.join(STATIC_FOLDER, "prototype_similarity")

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(GRADCAM_FOLDER, exist_ok=True)
os.makedirs(PROTOTYPE_OUTPUT_FOLDER, exist_ok=True)

HISTORY_FILE = "history.json"

# -------------------- Load Model ONCE --------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Loading multimodal DR model...")
model = load_model(device=device)
model.eval()

# 🔴 REQUIRED for Grad-CAM (disable inplace ReLU)
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

    # -------------------- Run Prediction --------------------
    try:
        result = run_prediction(image_path, age, hba1c, glucose_values)
        image_tensor, predicted_class = get_last_image_tensor_and_class()
    except Exception as e:
        return jsonify({"error": f"Prediction failed: {e}"}), 500

    # -------------------- Generate Patient ID --------------------
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

    # -------------------- Grad-CAM Generation --------------------
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

    print("✅ Grad-CAM saved:", gradcam_path)

    # -------------------- Prototype Similarity --------------------
    prototype_path = find_most_similar_prototype(
        uploaded_image_path=image_path,
        prototypes_root="prototypes",
        model=model,
        device=device
    )

    prototype_name = f"prototype_{patient_id}.jpg"
    prototype_save_path = os.path.join(PROTOTYPE_OUTPUT_FOLDER, prototype_name)

    shutil.copy(prototype_path, prototype_save_path)

    print("🧬 Prototype saved:", prototype_save_path)

    # -------------------- Lesion Report Safety --------------------
    lesion_info = generate_lesion_text(predicted_class)
    lesion_report = (
        f"{lesion_info['stage']}: " +
        ", ".join(lesion_info["lesions"])
    )
        

    # -------------------- Save History --------------------
    save_history({
        "patient_id": patient_id,
        "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "image_name": image.filename,
        "gradcam_image": f"/static/gradcam/{gradcam_filename}",
        "prototype_image": f"/static/prototype_similarity/{prototype_name}",
        "age": age,
        "hba1c": hba1c,
        "prediction_class": predicted_class,
        "prediction": result["prediction"],
        "confidence": round(result["confidence"] * 100, 2),
        "lesion_report": lesion_report
    })

    # -------------------- API Response --------------------
    result["patient_id"] = patient_id
    result["gradcam_image"] = f"/static/gradcam/{gradcam_filename}"
    result["prototype_image"] = f"/static/prototype_similarity/{prototype_name}"
    result["lesion_report"] = lesion_report

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
