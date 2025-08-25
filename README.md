# 🩺 Multi-Chronic Disease Detection Bot  

An AI-powered diagnostic support system that detects **multiple chronic respiratory diseases** from chest X-ray images using **Deep Learning + Machine Learning**, and connects patients to nearby doctors via an interactive chatbot.  

---

## 🚀 Project Overview  

The **Multi-Chronic Disease Detection Bot** is designed to assist in the **early diagnosis of respiratory illnesses** using chest X-ray images. It focuses on addressing challenges in **rural and under-resourced areas**, where access to specialized healthcare is often limited.  

The system detects **five major chronic respiratory diseases**:  
- 🫁 **Chronic Obstructive Pulmonary Disease (COPD)**  
- ❤️ **Pulmonary Hypertension**  
- 🦠 **Pneumonia**  
- 🫀 **Pulmonary Fibrosis**  
- 🧫 **Tuberculosis (TB)**  

---

## 🎯 Motivation  

Respiratory diseases are a **leading cause of morbidity worldwide**, and delayed diagnosis can lead to severe complications. Our bot bridges the gap by:  
- ✅ Providing **AI-powered automated detection**  
- ✅ Suggesting **nearby doctors** for treatment  
- ✅ Generating **personalized PDF medical reports**  

---

## 🧠 Methodology  

### 🔹 Data Collection & Preprocessing  
- Collected **publicly available chest X-ray datasets**  
- Applied preprocessing: resizing, grayscale conversion, normalization  
- Performed **data augmentation** for robustness  

### 🔹 Model Architecture  
- **DenseNet (Pre-trained CNN)** → Used as a **feature extractor**  
- **Random Forest Classifier** → Classifies extracted features into disease categories  
- Achieved **91.67% accuracy** on test data  

### 🔹 System Integration  
- **Flask Backend** → Manages disease detection & chatbot interface  
- **Chatbot** → Allows users to upload X-ray images & receive predictions  
- **GoMaps API** → Fetches **nearby doctors** with details (name, specialization, distance)  
- **PDF Report Generator** → Generates downloadable reports with patient details & predictions  

---

## 🛠️ Tech Stack  

- **Machine Learning & Deep Learning**: TensorFlow/Keras, scikit-learn  
- **Feature Extraction**: DenseNet  
- **Classifier**: Random Forest  
- **Web Framework**: Flask  
- **Frontend/Chatbot**: HTML, CSS, JS (via Flask integration)  
- **APIs**: GoMaps API (doctor suggestions)  
- **Report Generation**: ReportLab  

---

## ⚙️ System Workflow  

```mermaid
flowchart TD
A[Chest X-ray Image Upload] --> B[Preprocessing & Augmentation]  
B --> C[Feature Extraction with DenseNet]  
C --> D[Random Forest Classifier]  
D --> E[Predicted Disease Output]  
E --> F[Chatbot Interaction]  
F --> G[Nearby Doctors via GoMaps API]  
F --> H[PDF Report Generation]  
