from flask import Flask, request, jsonify
import requests
import os
import json
from datetime import datetime
app = Flask(__name__)
ML_SERVICE_URL = "http://127.0.0.1:8000"

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# -------------------------
# PREDICT (Frontend → Backend → ML)
# -------------------------
@app.route("/api/predict", methods=["POST"])
def predict():
    files = {"image": request.files["image"]}
    data = request.form.to_dict()

    response = requests.post(
        f"{ML_SERVICE_URL}/predict",
        files=files,
        data=data
    )

    return jsonify(response.json())

# -------------------------
# HISTORY (Frontend → Backend → ML)
# -------------------------
@app.route("/api/history", methods=["GET"])
def get_history():
    response = requests.get(f"{ML_SERVICE_URL}/history")
    return jsonify(response.json())


if __name__ == "__main__":
    app.run(port=5000, debug=True)
