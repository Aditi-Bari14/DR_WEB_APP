import requests

# 1️⃣ URL of your running ML service
url = "http://127.0.0.1:8000/predict"

# 2️⃣ Path to the test retina image
image_path = r"E:\DR_web_app\ml_service\uploads\PT_004992_R.jpg"

# 3️⃣ Open the image in binary mode
with open(image_path, "rb") as img_file:
    files = {"image": img_file}

    # 4️⃣ Tabular features: must match model input order and length (6 features)
    data = {
        "patient_id":1,
        "age": 54,             # integer
        "hba1c_latest": 7.2,   # float
        "glucose_mean": 120.0,
        "glucose_std": 10.0,
        "glucose_min": 100.0,
        "glucose_max": 140.0
    }

    # 5️⃣ Make POST request
    response = requests.post(url, files=files, data=data)

# 6️⃣ Print the JSON response
try:
    result = response.json()
    print("Prediction result:", result)
except Exception as e:
    print("Error decoding JSON:", e)
    print("Response text:", response.text)
