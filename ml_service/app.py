# app.py
from flask import Flask, request, jsonify
import os
from inference.predict import run_prediction  # real model inference

# -------------------- Flask app setup --------------------
app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# -------------------- Routes --------------------
# @app.route("/predict", methods=["POST"])
# def predict():
#     """
#     Accepts:
#         - image file
#         - age
#         - HbA1c
#     Returns:
#         - prediction (DR stage)
#         - confidence (float)
#     """
#     # 1️⃣ Check if image is uploaded
#     if "image" not in request.files:
#         return jsonify({"error": "No image uploaded"}), 400

#     image = request.files["image"]

#     # 2️⃣ Save image to uploads folder
#     image_path = os.path.join(UPLOAD_FOLDER, image.filename)
#     image.save(image_path)

#     # 3️⃣ Get metadata
#     try:
#         age = float(request.form.get("age"))
#         hba1c = float(request.form.get("hba1c"))
#         glucose_values = request.form["glucose_values"]
#     except (TypeError, ValueError):
#         return jsonify({"error": "Invalid or missing metadata"}), 400

#     # 4️⃣ Call REAL model prediction
#     # run_prediction expects image path/file and a list of tabular features
#     result = run_prediction(image_path, age, hba1c, glucose_values)

#     # 5️⃣ Return JSON response
#     return jsonify(result)


# @app.route("/", methods=["GET"])
# def home():
#     """Optional route to check if service is running"""
#     return jsonify({"message": "ML Service is running"}), 200

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
    except Exception as e:
        # Catch model errors and return as JSON
        return jsonify({"error": f"Prediction failed: {e}"}), 500

    return jsonify(result)

# -------------------- Main --------------------
if __name__ == "__main__":
    print("Starting ML service...")
    app.run(port=8000, debug=True)
