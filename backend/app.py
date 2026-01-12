from flask import Flask, request, jsonify
import requests
import os
import json
from datetime import datetime
app = Flask(__name__)
ML_SERVICE_URL = "http://127.0.0.1:8000"

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route("/api/history", methods=["GET"])
def get_history():
    try:
        with open("history.json", "r") as f:
            history = json.load(f)
    except FileNotFoundError:
        history = []

    history = sorted(history, key=lambda x: x["date"], reverse=True)
    return jsonify(history)

history_record = {
    "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "prediction": result["prediction"],
    "confidence": round(result["confidence"] * 100, 2)
}
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
    app.run(port=8000, debug=True)
