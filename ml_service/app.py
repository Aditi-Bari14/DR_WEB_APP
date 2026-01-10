from flask import Flask, request, jsonify
import os

app = Flask(__name__)

@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    image = request.files["image"]

    # TEMP result (replace later with real model prediction)
    result = {
        "prediction": "Moderate DR",
        "confidence": 0.92
    }

    return jsonify(result)

if __name__ == "__main__":
    print("Starting ML service...")
    app.run(port=8000, debug=True)
