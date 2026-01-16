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
@app.route("/explain/<patient_id>")
def explain(patient_id):
    response = requests.get(f"{BACKEND_URL}/api/history")
    history = response.json()

    record = next(r for r in history if r["patient_id"] == patient_id)

    gradcam_url = None
    if record.get("gradcam_image"):
        gradcam_url = ML_SERVICE_URL + record["gradcam_image"]

    return render_template(
        "explainability.html",
        record=record,
        gradcam_image=gradcam_url
    )

if __name__ == "__main__":
    app.run(port=3000, debug=True)
