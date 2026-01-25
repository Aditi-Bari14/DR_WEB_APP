from flask import Flask, request, jsonify
import requests
import os
from datetime import datetime
from models import db, PredictionHistory
from flask_migrate import Migrate
import json


app = Flask(__name__)

# Database config
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///database.db"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

db.init_app(app)


ML_SERVICE_URL = "http://127.0.0.1:8000"
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Create DB tables
with app.app_context():
    db.create_all()
@app.route("/api/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    image = request.files["image"]
    image_path = os.path.join(UPLOAD_FOLDER, image.filename)
    image.save(image_path)

    data = request.form.to_dict()
    files = {"image": open(image_path, "rb")}

    try:
        ml_response = requests.post(
            f"{ML_SERVICE_URL}/predict",
            files=files,
            data=data
        )
        print("STATUS:", ml_response.status_code)
        print("HEADERS:", ml_response.headers)
        print("RAW RESPONSE:", ml_response.text)
        ml_response.raise_for_status()  

        result = ml_response.json()

    except Exception as e:
        return jsonify({"error": f"ML service failed: {e}"}), 500

    record = PredictionHistory(
    patient_id=data["patient_id"],
    date=datetime.now(),
    image_name=image.filename,
    age=float(data["age"]),
    hba1c=float(data["hba1c"]),
    prediction=json.dumps(result["prediction"]),
    confidence=result["confidence"],
    gradcam_image=result["gradcam_image"],
    prototype_image=result["prototype_image"]
)


    db.session.add(record)
    db.session.commit()

    return jsonify(result)

#Get all history
@app.route("/api/history", methods=["GET"])
def get_history():
    records = PredictionHistory.query.order_by(
        PredictionHistory.date.desc()
    ).all()

    return jsonify([
        {
            "patient_id": r.patient_id,
            "date": r.date.strftime("%Y-%m-%d %H:%M:%S"),
            "image_name": r.image_name,
            "age": r.age,
            "hba1c": r.hba1c,
            "prediction": r.prediction,
            "confidence": r.confidence,
            "gradcam_image": r.gradcam_image,
            "prototype_image": r.prototype_image
        } for r in records
    ])

# PER-PATIENT HISTORY
@app.route("/api/history/<patient_id>", methods=["GET"])
def get_patient_history(patient_id):
    records = PredictionHistory.query.filter_by(
        patient_id=patient_id
    ).order_by(PredictionHistory.date.desc()).all()

    return jsonify([
        {
            "date": r.date.strftime("%Y-%m-%d %H:%M:%S"),
            "prediction": r.prediction,
            "confidence": r.confidence,
             "image_name": r.image_name,   # ✅ Add this
            "age": r.age,
            "hba1c": r.hba1c
        } for r in records
    ])
@app.route("/api/explain", methods=["POST"])
def explain():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    files = {"image": request.files["image"]}
    data = {"predicted_class": request.form.get("predicted_class")}

    try:
        response = requests.post(
            f"{ML_SERVICE_URL}/explain",
            files=files,
            data=data
        )
        response.raise_for_status()
        return jsonify(response.json())
    except Exception as e:
        return jsonify({"error": f"Explain failed: {e}"}), 500

# -------------------------
# Explainability
# -------------------------
# @app.route("/api/explain", methods=["POST"])
# def explain():
#     try:
#         image_path = request.json["image_path"]
#         predicted_class = request.json["predicted_class"]
#     except KeyError as e:
#         return jsonify({"error": f"Missing parameter: {e}"}), 400
    
#     if not image_path:
#         return jsonify({"error": "image_path missing"}), 400

#     if not os.path.isfile(image_path):
#         return jsonify({"error": f"Image not found at path: {image_path}"}), 400

#     try:
#         files = {"image": open(image_path, "rb")}        

#         data = {"predicted_class": predicted_class}

#         response = requests.post(
#                 f"{ML_SERVICE_URL}/explain",
#                 files=files,
#                 data=data
#             )

#         if response.status_code != 200:
#             return jsonify({
#                 "error": "ML explain failed",
#                 "details": response.text
#             }), 500

#         return jsonify(response.json())

#     except Exception as e:
#         return jsonify({"error": f"Explain request failed: {e}"}), 500

# -------------------------
# Main
# -------------------------
if __name__ == "__main__":
    app.run(port=5000, debug=True)
