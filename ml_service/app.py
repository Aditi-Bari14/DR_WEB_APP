
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

@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    image = request.files["image"]
    image_path = os.path.join(UPLOAD_FOLDER, image.filename)
    image.save(image_path)

    try:
        patient_id = request.form.get("patient_id")
        if not patient_id:
            raise ValueError("Patient ID missing")
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

    # -------------------- Lesion Report --------------------
    lesion_info = generate_lesion_text(predicted_class)
    lesion_report = (
        f"{lesion_info['stage']}: " +
        ", ".join(lesion_info["lesions"])
    )
    # -------------------- Final Response --------------------
    return jsonify({
        "prediction": result,
        "confidence": round(result["confidence"] * 100, 2),
        "predicted_class": int(predicted_class),
        "gradcam_image": f"/static/gradcam/{gradcam_filename}",
        "prototype_image": f"/static/prototype_similarity/{prototype_name}",
        "lesion_report": lesion_report
    })

    #return jsonify(result)

# @app.route("/explain", methods=["POST"])
# def explain():
#     if "image" not in request.files:
#         return jsonify({"error": "No image uploaded"}), 400

#     image = request.files["image"]
#     predicted_class = request.form.get("predicted_class")
#     if predicted_class is None:
#         return jsonify({"error": "predicted_class is required"}), 400

#     image_path = os.path.join(UPLOAD_FOLDER, image.filename)
#     image.save(image_path)

#     try:
#         # ------------------------
#         # Read real patient tabular data from request
#         # ------------------------
#         age = float(request.form.get("age", 0))
#         hba1c_latest = float(request.form.get("hba1c_latest", 0))
#         glucose_mean = float(request.form.get("glucose_mean", 0))
#         glucose_std = float(request.form.get("glucose_std", 0))
#         glucose_min = float(request.form.get("glucose_min", 0))
#         glucose_max = float(request.form.get("glucose_max", 0))

#         # Create tabular tensor [1, num_tab_features]
#         tabular_tensor = torch.tensor(
#             [[age, hba1c_latest, glucose_mean, glucose_std, glucose_min, glucose_max]],
#             dtype=torch.float
#         ).to(device)

#         # ------------------------
#         # Node index and graph features (use training graph)
#         # ------------------------
#         node_idx = torch.tensor([0], dtype=torch.long).to(device)  # map to first node
#         graph_feats = node_features.to(device)
#         edges = edge_index.to(device)

#         # ------------------------
#         # Preprocess image only
#         # ------------------------
#         image_tensor = preprocess(image_path, device=device)  # [1,C,H,W]

#         # ------------------------
#         # Grad-CAM
#         # ------------------------
#         cam_img = generate_gradcam(
#             model,
#             image_tensor,
#             int(predicted_class),
#             tabular_tensor=tabular_tensor,
#             graph_node_feats=graph_feats,
#             edge_index=edges,
#             node_idx=node_idx
#         )

#         # ------------------------
#         # Prototype explanation
#         # ------------------------
#         proto_info = proto_explanation(
#             model,
#             image_tensor,
#             tabular_tensor=tabular_tensor,
#             graph_node_feats=graph_feats,
#             edge_index=edges,
#             node_idx=node_idx
#         )

#         # ------------------------
#         # Save Grad-CAM image
#         # ------------------------
#         os.makedirs("static/explain", exist_ok=True)
#         cam_path = f"static/explain/cam_{image.filename}"
#         cv2.imwrite(cam_path, cv2.cvtColor(cam_img, cv2.COLOR_RGB2BGR))

#         return jsonify({
#             "cam_image": cam_path,
#             "prototype_info": proto_info
#         })

#     except Exception as e:
#         return jsonify({"error": str(e)}), 500

    # -------------------- Serve Uploads --------------------
@app.route("/uploads/<filename>")
def uploaded_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

# -------------------- Main --------------------
if __name__ == "__main__":
    print("Starting ML service...")
    app.run(port=8000, debug=True)


