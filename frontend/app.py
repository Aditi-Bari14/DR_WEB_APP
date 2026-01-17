from flask import Flask, render_template, request
import requests
import os

app = Flask(__name__)

BACKEND_URL = "http://127.0.0.1:5000"
ML_SERVICE_URL = "http://127.0.0.1:8000"  # ✅ NEW

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
# EXPLAINABILITY
# -------------------------
@app.route("/explainability")
def explainability():
    return render_template("explainability.html")

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
    data = {
        "age": request.form["age"],
        "hba1c": request.form["hba1c"],
        "glucose_values": request.form["glucose_values"]
    }

    response = requests.post(
        f"{BACKEND_URL}/api/predict",
        files=files,
        data=data
    )

    if response.status_code != 200:
        return f"Backend error: {response.text}", 500

    result = response.json()

    if "error" in result:
        return f"Prediction failed: {result['error']}", 500

    return render_template(
        "result.html",
        prediction=result["prediction"],
        confidence=result["confidence"]
    )

# -------------------------
# EXPLAIN (VIEW BUTTON)
# -------------------------
DR_CLASS_MAP = {
    0: "No DR",
    1: "Mild",
    2: "Moderate",
    3: "Severe",
    4: "Proliferative"
}
def generate_clinical_interpretation(prediction_label):
    interpretations = {
        "No DR": {
            "diagnosis": "No Diabetic Retinopathy",
            "findings": ["Normal retinal vasculature"],
            "meaning": "No visible signs of diabetic retinal damage.",
            "recommendation": "Routine annual eye examination recommended."
        },
        "Mild": {
            "diagnosis": "Mild Diabetic Retinopathy",
            "findings": ["Microaneurysms", "Subtle vascular changes"],
            "meaning": "Early-stage retinal damage due to diabetes.",
            "recommendation": "Regular monitoring and good glycemic control advised."
        },
        "Moderate": {
            "diagnosis": "Moderate Diabetic Retinopathy",
            "findings": ["Hemorrhages", "Microaneurysms", "Vascular abnormalities"],
            "meaning": "Progressive retinal damage with increased risk of vision impairment.",
            "recommendation": "Closer ophthalmic follow-up recommended."
        },
        "Severe": {
            "diagnosis": "Severe Diabetic Retinopathy",
            "findings": ["Widespread hemorrhages", "Venous abnormalities"],
            "meaning": "Advanced retinal ischemia with high risk of progression.",
            "recommendation": "Urgent ophthalmologist referral advised."
        },
        "Proliferative": {
            "diagnosis": "Proliferative Diabetic Retinopathy",
            "findings": ["Neovascularization", "Severe vascular proliferation"],
            "meaning": "Vision-threatening stage with abnormal new blood vessel growth.",
            "recommendation": "Immediate specialist intervention required."
        }
    }

    return interpretations.get(prediction_label, None)


@app.route("/explain/<patient_id>")
def explain(patient_id):
    response = requests.get(f"{BACKEND_URL}/api/history")
    history = response.json()

    record = next(r for r in history if r["patient_id"] == patient_id)

    gradcam_url = None
    if record.get("gradcam_image"):
        gradcam_url = ML_SERVICE_URL + record["gradcam_image"]

    # ✅ Generate interpretation from prediction
    prediction_class = record.get("prediction_class")  # 0–4 integer
    prediction_label = DR_CLASS_MAP.get(prediction_class)

    clinical_data = generate_clinical_interpretation(prediction_label)


    return render_template(
        "explainability.html",
        record=record,
        gradcam_image=gradcam_url,
        prototype_image=record.get("prototype_image"),
        clinical_data=clinical_data
    )



if __name__ == "__main__":
    app.run(port=3000, debug=True)
