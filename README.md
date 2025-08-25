🩺 Multi-Chronic Disease Detection Bot

An AI-powered diagnostic support system that detects multiple chronic respiratory diseases from chest X-ray images using Deep Learning + Machine Learning, and connects patients to nearby doctors via an interactive chatbot.

📌 Key Features

✅ Detects 5 chronic respiratory diseases (COPD, Pulmonary Hypertension, Pneumonia, Pulmonary Fibrosis, Tuberculosis)

✅ Achieved 91.67% model accuracy using CNN + Random Forest

✅ Interactive chatbot interface for real-time predictions

✅ Nearby doctors suggestion using GoMaps API

✅ Downloadable PDF medical reports with patient details & predictions

🎯 Motivation

Respiratory diseases are a leading cause of morbidity worldwide, and delayed diagnosis can lead to severe complications. Our bot bridges the gap by:

Providing AI-powered automated detection

Suggesting nearby doctors for treatment

Generating personalized medical reports

🧠 Methodology
🔹 Data Collection & Preprocessing

Used publicly available chest X-ray datasets

Preprocessing: resizing, grayscale conversion, normalization

Data augmentation for improved robustness

🔹 Model Architecture

DenseNet (Pre-trained CNN) → Used as feature extractor

Random Forest Classifier → Classifies extracted features into disease categories

Achieved 91.67% accuracy on test data

🔹 System Integration

Flask Backend → Handles detection & chatbot communication

Chatbot → Upload X-ray → Get prediction instantly

GoMaps API → Suggests nearby doctors with details

Report Generator → Generates downloadable PDF reports

⚙️ System Workflow
flowchart TD
A[Chest X-ray Image Upload] --> B[Preprocessing & Augmentation]  
B --> C[Feature Extraction with DenseNet]  
C --> D[Random Forest Classifier]  
D --> E[Predicted Disease Output]  
E --> F[Chatbot Interaction]  
F --> G[Nearby Doctors via GoMaps API]  
F --> H[PDF Report Generation]

📊 Results

🏆 91.67% test accuracy

🫁 Robust detection across five chronic respiratory diseases

💬 Successfully integrated chatbot + doctor suggestions + PDF reports

🛠️ Tech Stack

Deep Learning & ML: TensorFlow/Keras, scikit-learn

Feature Extraction: DenseNet

Classifier: Random Forest

Backend: Flask

Frontend: HTML, CSS, JS (Flask integration)

APIs: GoMaps API (doctor suggestions)

Reports: ReportLab (PDF generation)

📂 Project Structure
Multi-Chronic-Disease-Detection-Bot/
│── app/                  # Core Flask application  
│── frontend/             # Chatbot frontend (UI)  
│── models/               # Saved models & feature extractor  
│── notebooks/            # Jupyter notebooks for experiments  
│── static/               # Static files (CSS, JS, Images)  
│── templates/            # HTML templates for Flask  
│── reports/              # Generated PDF reports  
│── requirements.txt      # Dependencies  
│── README.md             # Documentation  

🖥️ How to Run

Clone the repository

git clone https://github.com/maddugarubhargavvijay/multi-chronic-disease-detection-bot.git
cd multi-chronic-disease-detection-bot


Create virtual environment & install dependencies

pip install -r requirements.txt


Run Flask app

python app/app.py


Access application
Open browser → http://127.0.0.1:5000/
