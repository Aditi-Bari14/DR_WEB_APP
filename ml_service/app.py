# app.py
from flask import Flask, request, jsonify
import os
from inference.predict import run_prediction  # real model inference
import json
from datetime import datetime

# -------------------- Flask app setup --------------------
app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# 🔹 NEW: history file
HISTORY_FILE = "history.json"


# 🔹 NEW: helper function to save history
def save_history(record):
    try:
        with open(HISTORY_FILE, "r") as f:
            history = json.load(f)
    except FileNotFoundError:
        history = []

    history.append(record)

    with open(HISTORY_FILE, "w") as f:
        json.dump(history, f, indent=4)


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
        return jsonify({"error": f"Prediction failed: {e}"}), 500

    # 🔹 NEW: save prediction history
    history_record = {
        "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "image_name": image.filename,
        "age": age,
        "hba1c": hba1c,
        "prediction": result["prediction"],
        "confidence": round(result["confidence"] * 100, 2)
    }

    save_history(history_record)

    return jsonify(result)


# 🔹 NEW: route to fetch history
@app.route("/history", methods=["GET"])
def get_history():
    try:
        with open(HISTORY_FILE, "r") as f:
            history = json.load(f)
    except FileNotFoundError:
        history = []
    # Optional: sort by date descending
    history = sorted(history, key=lambda x: x["date"], reverse=True)
    return jsonify(history)


# -------------------- Main --------------------
if __name__ == "__main__":
    print("Starting ML service...")
    app.run(port=8000, debug=True)
