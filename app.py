from flask import Flask, render_template, request, redirect, session
import mysql.connector
import numpy as np
import subprocess
import sys
import os
import pickle
from tensorflow.keras.models import load_model

# -------------------------------------------------
# APP CONFIG
# -------------------------------------------------
app = Flask(__name__)
app.secret_key = "secure_secret_key"

# -------------------------------------------------
# DATABASE CONNECTION
# -------------------------------------------------
db = mysql.connector.connect(
    host="localhost",
    user="root",
    password="root",      # change if needed
    database="alternative_medicine_db"
)
cursor = db.cursor()

# -------------------------------------------------
# PATHS
# -------------------------------------------------
MODEL_PATH = "model/cnn_model.h5"
ENCODER_PATH = "model/label_encoder.pkl"

# -------------------------------------------------
# LOAD MODEL & ENCODER
# -------------------------------------------------
model = None
label_encoder = None

if os.path.exists(MODEL_PATH) and os.path.exists(ENCODER_PATH):
    model = load_model(MODEL_PATH, compile=False)
    with open(ENCODER_PATH, "rb") as f:
        label_encoder = pickle.load(f)

# -------------------------------------------------
# HOME
# -------------------------------------------------
@app.route("/")
def index():
    return render_template("index.html")

# -------------------------------------------------
# ADMIN LOGIN
# -------------------------------------------------
@app.route("/admin", methods=["GET", "POST"])
def admin():
    if request.method == "POST":
        if request.form["username"] == "admin" and request.form["password"] == "admin":
            session["admin"] = True
            return redirect("/admin_dashboard")
    return render_template("admin_login.html")

# -------------------------------------------------
# ADMIN DASHBOARD
# -------------------------------------------------
@app.route("/admin_dashboard")
def admin_dashboard():
    if "admin" not in session:
        return redirect("/admin")
    return render_template("admin_dashboard.html")

# -------------------------------------------------
# TRAIN MODEL (ADMIN)
# -------------------------------------------------
@app.route("/train_model", methods=["POST"])
def train_model():
    if "admin" not in session:
        return redirect("/admin")

    # Run training script
    subprocess.run([sys.executable, "train_model.py"])

    # Reload model & encoder
    global model, label_encoder
    if os.path.exists(MODEL_PATH) and os.path.exists(ENCODER_PATH):
        model = load_model(MODEL_PATH, compile=False)
        with open(ENCODER_PATH, "rb") as f:
            label_encoder = pickle.load(f)

    return render_template(
        "admin_dashboard.html",
        message="Model trained successfully. Accuracy & graphs updated."
    )

# -------------------------------------------------
# USER REGISTRATION
# -------------------------------------------------
@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        cursor.execute(
            "INSERT INTO users(name,email,password) VALUES(%s,%s,%s)",
            (request.form["name"], request.form["email"], request.form["password"])
        )
        db.commit()
        return redirect("/login")
    return render_template("user_register.html")

# -------------------------------------------------
# USER LOGIN
# -------------------------------------------------
@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        cursor.execute(
            "SELECT * FROM users WHERE email=%s AND password=%s",
            (request.form["email"], request.form["password"])
        )
        user = cursor.fetchone()
        if user:
            session["user"] = user[0]
            return redirect("/predict")
    return render_template("user_login.html")

# -------------------------------------------------
# PREDICTION
# -------------------------------------------------
@app.route("/predict", methods=["GET", "POST"])
def predict():
    if "user" not in session:
        return redirect("/login")

    if model is None or label_encoder is None:
        return "Model not trained yet. Please contact Admin."

    if request.method == "POST":

        # EXACT FEATURE ORDER (MATCHES TRAINING)
        feature_vector = [
            int(request.form["Age_Group"]),
            int(request.form["Gender"]),
            int(request.form["Primary_Symptom"]),
            int(request.form["Secondary_Symptom"]),
            int(request.form["Symptom_Duration_Days"]),
            int(request.form["Severity"]),
            int(request.form["Chronic_Condition"]),
            int(request.form["Stress_Level"]),
            int(request.form["Sleep_Quality"]),
            int(request.form["Lifestyle"]),
            int(request.form["Diet_Type"]),
            int(request.form["Previous_Treatment"]),
            int(request.form["Dosage_Form"]),
            int(request.form["Treatment_Duration"]),
            int(request.form["Follow_Up_Required"])
        ]

        # create 110 feature vector
        data = np.zeros(110)

        # fill first 15 positions with form values
        data[:15] = feature_vector
        # reshape for CNN
        data = data.reshape(1, 110, 1)

        prediction = model.predict(data)
        pred_class = np.argmax(prediction)
        confidence = np.max(prediction) * 100

        # CORRECT label decoding
        result = label_encoder.inverse_transform([pred_class])[0]

        if confidence < 60:
            result = "Consult Practitioner (Low Confidence Prediction)"

        cursor.execute(
            "INSERT INTO predictions(user_id,primary_symptom,severity,result) VALUES(%s,%s,%s,%s)",
            (
                session["user"],
                request.form["Primary_Symptom"],
                request.form["Severity"],
                result
            )
        )
        db.commit()

        return render_template(
            "result.html",
            result=result,
            confidence=round(confidence, 2)
        )

    return render_template("predict.html")

# -------------------------------------------------
# LOGOUT
# -------------------------------------------------
@app.route("/logout")
def logout():
    session.clear()
    return redirect("/")

# -------------------------------------------------
# RUN
# -------------------------------------------------
if __name__ == "__main__":
    app.run(debug=True)
