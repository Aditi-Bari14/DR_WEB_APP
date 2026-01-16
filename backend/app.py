from flask import Flask, request, jsonify
import requests
import os

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

    # ✅ HANDLE ML ERRORS SAFELY
    if response.status_code != 200:
        return jsonify({
            "error": "ML service failed",
            "details": response.text
        }), 500

    # ✅ SAFE JSON PARSE
    try:
        return jsonify(response.json())
    except Exception as e:
        return jsonify({
            "error": "Invalid JSON from ML service",
            "details": str(e)
        }), 500


# -------------------------
# HISTORY
# -------------------------
@app.route("/api/history", methods=["GET"])
def get_history():
    response = requests.get(f"{ML_SERVICE_URL}/history")

    if response.status_code != 200:
        return jsonify([])

    return jsonify(response.json())


if __name__ == "__main__":
    app.run(port=5000, debug=True)
