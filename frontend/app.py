from flask import Flask, render_template, request, redirect, url_for
import requests
import os

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
@app.route("/explainability")
def explainability():
    return render_template("explainability.html")

# <!-- Change made on Aditi branch -->

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

    result = response.json()

    return render_template(
        "result.html",
        prediction=result["prediction"],
        confidence=result["confidence"]
    )

if __name__ == "__main__":
    app.run(port=3000, debug=True)
