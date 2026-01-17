from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()

class PredictionHistory(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    patient_id = db.Column(db.String(50), nullable=False)
    date = db.Column(db.DateTime, nullable=False)
    image_name = db.Column(db.String(100))
    age = db.Column(db.Float)
    hba1c = db.Column(db.Float)
    prediction = db.Column(db.String(50))
    confidence = db.Column(db.Float)
