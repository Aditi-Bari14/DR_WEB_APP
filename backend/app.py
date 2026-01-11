from flask import Flask, request, jsonify
import os

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route("/predict", methods=["POST"])
def predict():
    # 1. Get inputs
    image = request.files["image"]
    age = request.form["age"]
    hba1c = request.form["hba1c"]

    # 2. Save image
    image_path = os.path.join(UPLOAD_FOLDER, image.filename)
    image.save(image_path)

    # 3. Dummy prediction (for now)
    prediction = "Moderate Diabetic Retinopathy"
    confidence = 0.92

    # 4. Return JSON
    return jsonify({
        "prediction": prediction,
        "confidence": confidence
    })

if __name__ == "__main__":
    app.run(port=8000, debug=True)
