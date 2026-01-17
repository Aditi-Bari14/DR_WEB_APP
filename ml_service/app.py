
from flask import Flask, request, jsonify
import os
from inference.predict import run_prediction  # real model inference
from datetime import datetime
from explainability.gradcam import generate_gradcam
from explainability.proto_explain import proto_explanation
# Import your preprocessing and model
from inference.predict import model, device
from inference.preprocess import preprocess
import cv2
# -------------------- Flask app setup --------------------
app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


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
    except Exception as e:
        return jsonify({"error": f"Prediction failed: {e}"}), 500



    return jsonify(result)

@app.route("/explain", methods=["POST"])
def explain():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    image = request.files["image"]
    predicted_class = request.form.get("predicted_class")
    if predicted_class is None:
        return jsonify({"error": "predicted_class is required"}), 400

    image_path = os.path.join(UPLOAD_FOLDER, image.filename)
    image.save(image_path)

    try:
        # ------------------------
        # Read real patient tabular data from request
        # ------------------------
        age = float(request.form.get("age", 0))
        hba1c_latest = float(request.form.get("hba1c_latest", 0))
        glucose_mean = float(request.form.get("glucose_mean", 0))
        glucose_std = float(request.form.get("glucose_std", 0))
        glucose_min = float(request.form.get("glucose_min", 0))
        glucose_max = float(request.form.get("glucose_max", 0))

        # Create tabular tensor [1, num_tab_features]
        tabular_tensor = torch.tensor(
            [[age, hba1c_latest, glucose_mean, glucose_std, glucose_min, glucose_max]],
            dtype=torch.float
        ).to(device)

        # ------------------------
        # Node index and graph features (use training graph)
        # ------------------------
        node_idx = torch.tensor([0], dtype=torch.long).to(device)  # map to first node
        graph_feats = node_features.to(device)
        edges = edge_index.to(device)

        # ------------------------
        # Preprocess image only
        # ------------------------
        image_tensor = preprocess(image_path, device=device)  # [1,C,H,W]

        # ------------------------
        # Grad-CAM
        # ------------------------
        cam_img = generate_gradcam(
            model,
            image_tensor,
            int(predicted_class),
            tabular_tensor=tabular_tensor,
            graph_node_feats=graph_feats,
            edge_index=edges,
            node_idx=node_idx
        )

        # ------------------------
        # Prototype explanation
        # ------------------------
        proto_info = proto_explanation(
            model,
            image_tensor,
            tabular_tensor=tabular_tensor,
            graph_node_feats=graph_feats,
            edge_index=edges,
            node_idx=node_idx
        )

        # ------------------------
        # Save Grad-CAM image
        # ------------------------
        os.makedirs("static/explain", exist_ok=True)
        cam_path = f"static/explain/cam_{image.filename}"
        cv2.imwrite(cam_path, cv2.cvtColor(cam_img, cv2.COLOR_RGB2BGR))

        return jsonify({
            "cam_image": cam_path,
            "prototype_info": proto_info
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# # Explainability
# @app.route("/explain", methods=["POST"])
# def explain():
#     image_path = request.files.get("image_path")
#     if not image_path or not os.path.isfile(image_path):
#         return jsonify({"error": f"Image not found at path: {image_path}"}), 400

#     predicted_class = request.form.get("predicted_class")
    
#     try:
#         # dummy tabular data (needed for your multimodal model)
#         tabular_dummy = {
#             "age": 0,
#             "hba1c_latest": 0,
#             "glucose_values": "0,0,0"
#         }

#         image_tensor, _ = preprocess(image_path, tabular_dummy, device=device)
#         cam_img = generate_gradcam(model, image_tensor, predicted_class)
#         proto_info = proto_explanation(model, image_tensor)

#         save_path = f"static/explain/cam_{os.path.basename(image_path)}"
#         os.makedirs(os.path.dirname(save_path), exist_ok=True)
#         cv2.imwrite(save_path, cv2.cvtColor(cam_img, cv2.COLOR_RGB2BGR))

#         return jsonify({
#             "cam_image": save_path,
#             "prototype_info": proto_info
#         })

#     except Exception as e:
#         return jsonify({"error": str(e)}), 500

# -------------------- Main --------------------
if __name__ == "__main__":
    print("Starting ML service...")
    app.run(port=8000, debug=True)


