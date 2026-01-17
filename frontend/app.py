from flask import Flask, render_template, request, redirect, url_for
import requests
import os
import matplotlib
matplotlib.use('Agg')   # ✅ NON-GUI backend

import matplotlib.pyplot as plt
import io
import base64
from datetime import datetime

app = Flask(__name__)
BACKEND_URL = "http://127.0.0.1:5000"

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# -------------------------
# LANDING PAGE
# -------------------------
@app.route("/")
def home():
    return render_template("index.html")

# -------------------------
# LOGIN & REGISTER
# -------------------------
@app.route("/login")
def login():
    return render_template("login.html")

@app.route("/register")
def register():
    return render_template("register.html")

# -------------------------
# DASHBOARD
# -------------------------
@app.route("/dashboard")
def dashboard():
    return render_template("dashboard.html")

# -------------------------
# HISTORY
# -------------------------
@app.route("/view-history")
def view_history():
    response = requests.get(f"{BACKEND_URL}/api/history")
    history = response.json()
    return render_template("history.html", history=history)


# -------------------------
# EXPLAINABILITY (XAI)


# -------------------------
# ADMIN
# -------------------------
@app.route("/admin")
def admin():
    return render_template("admin.html")

@app.route("/predict-dr")
def predict_dr_page():
    return render_template("predict.html")


# -------------------------
# PREDICTION
# -------------------------
@app.route("/predict", methods=["POST"])
def predict():
    image = request.files["image"]
    image_path = os.path.join(UPLOAD_FOLDER, image.filename)
    image.save(image_path)

    files = {"image": open(image_path, "rb")}
    patient_id = request.form["patient_id"]
    data = {
        "patient_id": patient_id,
        "age": request.form["age"],
        "hba1c": request.form["hba1c"],
        "glucose_values": request.form["glucose_values"]
    }

    try:
        response = requests.post(
            f"{BACKEND_URL}/api/predict",
            files=files,
            data=data
        )
        result = response.json()
    except Exception as e:
        result = {"error": f"Backend request failed: {e}"}

    # ✅ Safe check for errors
    if "error" in result:
        return f"Error: {result['error']}"

    prediction = result.get("prediction", "N/A")
    confidence = result.get("confidence", 0)

    return render_template(
        "result.html",
        prediction=prediction,
        confidence=confidence,
        patient_id=patient_id
    )

def generate_history_plot(history):
    """
    Generate a timeline plot for patient DR predictions
    """
    if not history:
        return None

    # Sort by date ascending
    history_sorted = sorted(history, key=lambda x: x["date"])

    # Extract dates and predictions
    dates = [datetime.strptime(r["date"], "%Y-%m-%d %H:%M:%S").strftime("%Y-%m-%d") for r in history_sorted]

    predictions = [r["prediction"] for r in history_sorted]

    # Map DR classes to numeric values for plotting
    class_order = {"No DR":0, "Mild":1, "Moderate":2, "Severe":3, "Proliferative DR":4}
    y_values = [class_order.get(p, -1) for p in predictions]

    # Create plot
    plt.figure(figsize=(8,4))
    plt.plot(dates, y_values, marker='o', linestyle='-', color='blue')
    plt.yticks(list(class_order.values()), list(class_order.keys()))
    plt.xlabel("Date")
    plt.ylabel("DR Severity")
    plt.title("Patient DR History Over Time")
    plt.grid(True, linestyle='--', alpha=0.5)

    # Save plot to PNG in memory
    buf = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format='png')
    plt.close()
    buf.seek(0)

    # Encode as base64 to embed in HTML
    img_base64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return img_base64

@app.route("/patient/<patient_id>")
def patient_history(patient_id):
    response = requests.get(f"{BACKEND_URL}/api/history/{patient_id}")
    try:
        history = response.json()
    except Exception as e:
        history = []

    plot_img = generate_history_plot(history)

    return render_template(
        "patient_history.html",
        history=history,
        patient_id=patient_id,
        plot_img=plot_img
    )
@app.route("/explain/<patient_id>")
def explain_page(patient_id):
    # Get latest patient record
    response = requests.get(f"{BACKEND_URL}/api/history/{patient_id}")
    if response.status_code != 200:
        return f"Failed to fetch history: {response.text}", 500
    history = response.json()
    if not history:
        return "No record found", 404

    record = history[0]  # latest prediction

       # 3️⃣ Open the image
    image_path = os.path.join("uploads", record["image_name"])
    if not os.path.exists(image_path):
        return f"Image not found: {image_path}", 404
    
    # Map prediction to numeric before sending
    class_map = {"No DR":0, "Mild":1, "Moderate":2, "Severe":3, "Proliferative DR":4}
    pred_class_idx = class_map.get(record["prediction"], 0)


    files = {"image": open(image_path, "rb")}
    data = {"predicted_class": str(pred_class_idx)}

    explain_resp = requests.post(
        f"{BACKEND_URL}/api/explain",
        files=files,
        data=data
    )
    if explain_resp.status_code != 200:
        return f"Explainability failed: {explain_resp.text}", 500


    result = explain_resp.json()

    proto_info = result.get("prototype_info", {})
    top_proto = proto_info.get("top_prototypes", [])
    scores = proto_info.get("similarity_scores", [])

    return render_template(
        "explainability.html",
        cam_image=result.get("cam_image"),
        proto_ids=top_proto,
        proto_scores=scores,
        age=record.get("age", 0),
        hba1c=record.get("hba1c", 0),
        glucose_mean=record.get("glucose_mean", 0),
        glucose_std=record.get("glucose_std", 0),
        patient_id=patient_id
    )

# Explainability page
# @app.route("/explain/<patient_id>")
# def explain_page(patient_id):
#     # 1️⃣ Fetch full history
#     resp = requests.get(f"{BACKEND_URL}/api/history")
#     if resp.status_code != 200:
#         return "Failed to fetch history", 500

#     history = resp.json()

#     # 2️⃣ Filter record for this patient
#     record = next((r for r in history if str(r["patient_id"]) == str(patient_id)), None)
#     if not record:
#         return "No record found for this patient", 404

#     files = {"image": open(f"uploads/{record['image_name']}", "rb")}
#     data = {"predicted_class": record["prediction"]}
#     explain_resp = requests.post(
#         f"{BACKEND_URL}/api/explain",
#         explain_resp = requests.post(f"{BACKEND_URL}/api/explain", files=files, data=data)

#     )

#     if explain_resp.status_code != 200:
#         return f"Explainability failed: {explain_resp.text}", 500

#     explain_data = explain_resp.json()

#     # 4️⃣ Extract prototype info
#     proto_info = explain_data.get("prototype_info", {})
#     top_proto = proto_info.get("top_prototypes", [])
#     scores = proto_info.get("similarity_scores", [])

#     # 5️⃣ Render explainability template
#     return render_template(
#         "explainability.html",  # your template
#         cam=explain_data["cam_image"],
#         proto_ids=top_proto,
#         proto_scores=scores,
#         age=record["age"],
#         hba1c=record["hba1c"],
#         glucose_mean=record.get("glucose_mean", 0),
#         glucose_std=record.get("glucose_std", 0),
#         patient_id=record["patient_id"]
#     )



if __name__ == "__main__":
    app.run(port=3000, debug=True)
