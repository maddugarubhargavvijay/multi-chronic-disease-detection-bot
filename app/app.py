from flask import Flask, request, jsonify, send_file
import tensorflow as tf
import numpy as np
import cv2
import requests
from fpdf import FPDF
from io import BytesIO
from flask_cors import CORS
from fpdf.enums import XPos, YPos
import os
import joblib  # to load RF model
from tensorflow.keras.models import Model
app = Flask(__name__)
CORS(app)

# ✅ Load CNN Model
# ✅ Load CNN Model (feature extractor)
cnn_model = tf.keras.models.load_model(r"F:\Multi_Chronic_Disease_Detection\model\densenet_new_finetuned_v3.h5")

# ✅ Load RF Classifier
rf = joblib.load(r"F:\Multi_Chronic_Disease_Detection\notebooks\fast_rf_xgb_stack2.pkl")

uploaded_xray_path = None

# ✅ Google Maps API Key (Replace with a **VALID** API Key)
GOMAPS_API_KEY = "AlzaSyWgwyvQn11Qsm6c49MSsbDJdFvUSqVqj6u"

# ✅ Define disease classes
disease_classes = ["COPD", "fibrosis", "normal", "pneumonia", "pulmonary tb"]

# ✅ Store user-selected doctor
user_selected_doctor = {}

# ✅ Process X-ray image
def preprocess_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (224, 224))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    return img

# ✅ Predict Disease
@app.route("/predict", methods=["POST"])
def predict():
    global uploaded_xray_path

    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    os.makedirs("uploads", exist_ok=True)
    uploaded_xray_path = os.path.join("uploads", file.filename)
    file.save(uploaded_xray_path)

    # ✅ Preprocess image
    img = preprocess_image(uploaded_xray_path)

    # ✅ Extract features using CNN (remove final Dense if needed)
    feature_extractor = Model(inputs=cnn_model.input, outputs=cnn_model.get_layer("global_average_pooling2d").output)
    features = feature_extractor.predict(img)  # shape: (1, 1024) or similar
    features = features.reshape(1, -1)  # flatten for RF input

    # ✅ Predict with RF
    rf_prediction = rf.predict(features)[0]
    rf_probabilities = rf.predict_proba(features)[0]
    confidence = max(rf_probabilities) * 100  # highest probability

    predicted_disease = disease_classes[int(rf_prediction)]

    return jsonify({
        "disease": predicted_disease,
        "confidence": round(confidence, 2),
        "image_path": uploaded_xray_path
    })

#  Get Nearby Doctors
@app.route("/get_doctors", methods=["POST"])
def get_doctors():
    try:
        latitude = request.json.get("latitude")
        longitude = request.json.get("longitude")

        if not latitude or not longitude:
            return jsonify({"error": "Location not provided"}), 400

        url = "https://maps.gomaps.pro/maps/api/place/nearbysearch/json"
        params = {
            "location": f"{latitude},{longitude}",
            "radius": 50000,
            "key": GOMAPS_API_KEY,
            "keyword": "pulmonologist"
        }

        response = requests.get(url, params=params, verify=False)  #  Added `verify=False`
        results = response.json().get("results", [])

        doctors = [
            {"name": place.get("name"), "location": place.get("vicinity", "Unknown Location")}
            for place in results
        ]

        if not doctors:
            return jsonify({"error": "No doctors found nearby"}), 404

        # Store available doctors in global variable
        user_selected_doctor["doctors"] = doctors

        #  Return numbered list of doctors
        doctor_list = {str(i+1): doctor for i, doctor in enumerate(doctors)}
        return jsonify({"doctors": doctor_list})

    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"error": "Failed to fetch doctors"}), 500

# Select a Doctor
@app.route("/select_doctor", methods=["POST"])
def select_doctor():
    try:
        doctor_index = request.json.get("doctor_index")

        if not doctor_index or "doctors" not in user_selected_doctor:
            return jsonify({"error": "Invalid doctor selection"}), 400

        doctors = user_selected_doctor["doctors"]
        doctor_index = int(doctor_index) - 1  # Convert to zero-based index

        if doctor_index < 0 or doctor_index >= len(doctors):
            return jsonify({"error": "Doctor not found"}), 404

        selected_doctor = doctors[doctor_index]
        user_selected_doctor["selected"] = selected_doctor

        return jsonify({
            "message": f"Doctor {selected_doctor['name']} selected successfully",
            "doctor": selected_doctor
        })

    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"error": "Failed to select doctor"}), 500

# PDF Report Class
class PDF(FPDF):
    def header(self):
        self.set_font("Arial", 'B', 14)
        self.cell(200, 10, "Health Report", new_x=XPos.LMARGIN, new_y=YPos.NEXT, align='C')
        self.ln(10)

# Generate Report
@app.route('/generate_report', methods=['POST'])
def generate_report():
    try:
        global uploaded_xray_path

        data = request.json
        user_name = data.get("user_name", "Unknown")
        disease = data.get("disease", "No prediction")
        chat_history = data.get("chat_history", [])
        selected_doctor = user_selected_doctor.get("selected", {"name": "None", "location": "Not selected"})

        pdf = PDF()
        pdf.add_page()
        pdf.set_font("Helvetica", size=12)  # ✅ Fix Arial warning

        pdf.cell(200, 10, f"Patient Name: {user_name}", new_x=XPos.LMARGIN, new_y=YPos.NEXT, align='L')
        pdf.cell(200, 10, f"Predicted Disease: {disease}", new_x=XPos.LMARGIN, new_y=YPos.NEXT, align='L')
        pdf.ln(10)

        pdf.cell(200, 10, "Chat History:", new_x=XPos.LMARGIN, new_y=YPos.NEXT, align='L')
        for chat in chat_history:
            pdf.multi_cell(180, 10, f"{chat['sender']}: {chat['text']}")
            pdf.ln(2)  # Adds a small space between messages


        pdf.ln(10)
        pdf.cell(200, 10, "Selected Doctor:", new_x=XPos.LMARGIN, new_y=YPos.NEXT, align='L')
        pdf.cell(200, 10, f"Doctor: {selected_doctor['name']}", new_x=XPos.LMARGIN, new_y=YPos.NEXT, align='L')
        pdf.cell(200, 10, f"Location: {selected_doctor['location']}", new_x=XPos.LMARGIN, new_y=YPos.NEXT, align='L')

        # ✅ Check if X-ray Exists Before Adding to PDF
        if uploaded_xray_path and os.path.exists(uploaded_xray_path):
            print("✅ X-ray image found, adding to PDF")
            pdf.ln(10)
            pdf.cell(200, 10, "X-ray Image:", new_x=XPos.LMARGIN, new_y=YPos.NEXT, align='L')
            pdf.image(uploaded_xray_path, x=50, y=None, w=100, h=100)

        # ✅ Save PDF File
        pdf_filename = "health_report.pdf"
        pdf.output(pdf_filename)

        # ✅ Send File with Proper Download Name
        response = send_file(pdf_filename, as_attachment=True, download_name="Health_Report.pdf", mimetype="application/pdf")

        # ✅ Cleanup after sending response
        if uploaded_xray_path and os.path.exists(uploaded_xray_path):
            os.remove(uploaded_xray_path)  # Delete uploaded image

        return response
    
    except Exception as e:
        print("❌ Error generating report:", e)
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
