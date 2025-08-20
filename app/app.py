from flask import Flask, request, jsonify, send_file, render_template_string, session
import tensorflow as tf
import numpy as np
import cv2
from flask_cors import CORS
import requests
from fpdf import FPDF
from fpdf.enums import XPos, YPos
import os
import joblib
from tensorflow.keras.models import Model
import logging

from requests.packages.urllib3.exceptions import InsecureRequestWarning
requests.packages.urllib3.disable_warnings(InsecureRequestWarning)

logging.basicConfig(level=logging.INFO)

app = Flask(__name__)
app.secret_key = 'your_super_secret_key'
CORS(app)
# -------------------------------
# CONFIGURATION
# -------------------------------
# -------------------------------
# CONFIGURATION
# -------------------------------
try:
    # Base directory where this file is running
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    # Move up one level to repo root
    ROOT_DIR = os.path.abspath(os.path.join(BASE_DIR, ".."))

    # Model paths
    cnn_model_path = os.path.join(ROOT_DIR, "models", "densenet_new_finetuned_v3.h5")
    rf_model_path = os.path.join(ROOT_DIR, "models", "fast_rf_xgb_stack2.pkl")

    # Load models
    cnn_model = tf.keras.models.load_model(cnn_model_path)
    rf_model = joblib.load(rf_model_path)

    # Create feature extractor from CNN
    feature_extractor = Model(
        inputs=cnn_model.input,
        outputs=cnn_model.get_layer("global_average_pooling2d").output
    )

    logging.info("✅ Models loaded successfully")

except Exception as e:
    logging.error(f"❌ Error loading models: {e}")
    cnn_model = rf_model = feature_extractor = None


GOMAPS_API_KEY = "YOUR_GOMAPS_API_KEY"
DISEASE_CLASSES = ["COPD", "fibrosis", "normal", "pneumonia", "pulmonary tb"]

# -------------------------------
# HTML Template with Advanced UI (No Emojis)
# -------------------------------
chatbot_html = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Multi-Chronic Disease Detection Chatbot</title>
<style>
@import url('https://fonts.googleapis.com/css2?family=Roboto:wght@400;500;700&display=swap');

:root {
    --primary: #667eea;
    --secondary: #764ba2;
    --tertiary: #f2f4f8;
    --card-bg: rgba(255,255,255,0.92);
    --bot-bg: rgba(255,255,255,0.88);
    --user-bg: #764ba2;
    --border-radius: 22px;
    --indicator-shadow: 0 2px 8px rgba(0,0,0,0.15);
}

body {
    font-family: 'Roboto', sans-serif;
    margin: 0;
    min-height: 100vh;
    display: flex;
    justify-content: center;
    align-items: center;
    background: linear-gradient(135deg,#667eea,#764ba2 60%, #f2f4f8 100%);
    position: relative;
    overflow: hidden;
}

/* Floating Background Elements */
.floating-elements {
    position: fixed;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    pointer-events: none;
    z-index: 0;
}

.floating-circle {
    position: absolute;
    border-radius: 50%;
    background: rgba(255, 255, 255, 0.1);
    animation: float 15s infinite ease-in-out;
}

.floating-circle:nth-child(1) {
    width: 80px;
    height: 80px;
    top: 20%;
    left: 10%;
    animation-delay: 0s;
    animation-duration: 20s;
}

.floating-circle:nth-child(2) {
    width: 120px;
    height: 120px;
    top: 60%;
    right: 15%;
    animation-delay: -5s;
    animation-duration: 25s;
}

.floating-circle:nth-child(3) {
    width: 60px;
    height: 60px;
    top: 80%;
    left: 20%;
    animation-delay: -10s;
    animation-duration: 18s;
}

.floating-circle:nth-child(4) {
    width: 100px;
    height: 100px;
    top: 30%;
    right: 30%;
    animation-delay: -15s;
    animation-duration: 22s;
}

.floating-circle:nth-child(5) {
    width: 70px;
    height: 70px;
    top: 10%;
    right: 50%;
    animation-delay: -8s;
    animation-duration: 19s;
}

@keyframes float {
    0%, 100% { transform: translateY(0px) rotate(0deg); }
    25% { transform: translateY(-20px) rotate(90deg); }
    50% { transform: translateY(-40px) rotate(180deg); }
    75% { transform: translateY(-20px) rotate(270deg); }
}

/* Floating Medical Icons */
.medical-icons {
    position: fixed;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    pointer-events: none;
    z-index: 0;
}

.medical-icon {
    position: absolute;
    font-size: 24px;
    opacity: 0.1;
    color: white;
    animation: floatIcon 20s infinite ease-in-out;
}

.medical-icon:nth-child(1) {
    top: 15%;
    left: 5%;
    animation-delay: 0s;
}

.medical-icon:nth-child(2) {
    top: 70%;
    right: 10%;
    animation-delay: -7s;
}

.medical-icon:nth-child(3) {
    top: 45%;
    left: 8%;
    animation-delay: -14s;
}

.medical-icon:nth-child(4) {
    top: 25%;
    right: 25%;
    animation-delay: -3s;
}

@keyframes floatIcon {
    0%, 100% { transform: translateY(0px) translateX(0px); opacity: 0.1; }
    25% { transform: translateY(-30px) translateX(10px); opacity: 0.2; }
    50% { transform: translateY(-60px) translateX(-10px); opacity: 0.15; }
    75% { transform: translateY(-30px) translateX(5px); opacity: 0.1; }
}

.container {
    width: 100%;
    max-width: 560px;
    height: 90vh;
    background: rgba(255,255,255,0.13);
    backdrop-filter: blur(25px);
    border-radius: var(--border-radius);
    box-shadow: 0 25px 50px rgba(0,0,0,0.15), 0 0 0 1px rgba(255,255,255,0.2);
    display: flex;
    flex-direction: column;
    overflow: hidden;
    border: 1px solid rgba(255,255,255,0.22);
    position: relative;
    z-index: 1;
    animation: containerFloat 8s ease-in-out infinite;
}

@keyframes containerFloat {
    0%, 100% { transform: translateY(0px); }
    50% { transform: translateY(-10px); }
}

.chat-header {
    background: linear-gradient(135deg, #667eea, #764ba2 80%);
    color: #fff;
    padding: 24px;
    font-size: 1.42rem;
    font-weight: 700;
    text-align: center;
    letter-spacing: 1.2px;
    box-shadow: 0 1px 10px rgba(118, 75, 162, 0.08);
    border-bottom: 1px solid rgba(255,255,255,0.15);
    position: relative;
    overflow: hidden;
}

.chat-header::before {
    content: '';
    position: absolute;
    top: 0;
    left: -100%;
    width: 100%;
    height: 100%;
    background: linear-gradient(90deg, transparent, rgba(255,255,255,0.2), transparent);
    animation: headerShine 3s infinite;
}

@keyframes headerShine {
    0% { left: -100%; }
    100% { left: 100%; }
}

.status-dot {
    position: absolute;
    top: 20px;
    right: 20px;
    width: 12px;
    height: 12px;
    background: #27ae60;
    border-radius: 50%;
    box-shadow: 0 0 0 4px rgba(39, 174, 96, 0.3);
    animation: pulse 2s infinite;
}

@keyframes pulse {
    0% { box-shadow: 0 0 0 0 rgba(39, 174, 96, 0.7); }
    70% { box-shadow: 0 0 0 10px rgba(39, 174, 96, 0); }
    100% { box-shadow: 0 0 0 0 rgba(39, 174, 96, 0); }
}

.chat-window {
    flex-grow: 1;
    padding: 24px 20px;
    overflow-y: auto;
    display: flex;
    flex-direction: column;
    gap: 8px;
    background: none;
    position: relative;
}

.chat-window::-webkit-scrollbar {
    width: 8px;
}
.chat-window::-webkit-scrollbar-thumb {
    background: rgba(102, 126, 234, 0.14);
    border-radius: 5px;
}

.message {
    max-width: 82%;
    padding: 16px 20px;
    border-radius: 18px;
    margin-bottom: 12px;
    line-height: 1.52;
    word-wrap: break-word;
    background: var(--card-bg);
    box-shadow: 0 2px 12px rgba(130,140,153,0.06);
    animation: messageFloat 0.6s cubic-bezier(0.4,0,0.2,1);
    font-size: 1.05rem;
    position: relative;
    transition: transform 0.3s ease, box-shadow 0.3s ease;
}

.message:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 25px rgba(130,140,153,0.12);
}

@keyframes messageFloat {
    0% { 
        opacity: 0; 
        transform: translateY(30px) scale(0.9);
    }
    50% {
        opacity: 0.8;
        transform: translateY(-5px) scale(1.02);
    }
    100% { 
        opacity: 1; 
        transform: translateY(0) scale(1);
    }
}

.bot-message {
    background: var(--bot-bg);
    color: #212941;
    align-self: flex-start;
    box-shadow: 0 2px 10px rgba(118, 75, 162, 0.06);
    border-left: 3px solid rgba(118, 75, 162, 0.3);
}

.bot-message strong { color: #764ba2; }

.user-message {
    background: var(--user-bg);
    color: #fff;
    align-self: flex-end;
    box-shadow: 0 2px 10px rgba(102, 126, 234, 0.10);
    border-right: 3px solid rgba(255, 255, 255, 0.3);
}

.disease-indicator {
    display: inline-block;
    width: 15px;
    height: 15px;
    border-radius: 50%;
    margin-right: 8px;
    vertical-align: middle;
    box-shadow: var(--indicator-shadow);
    position: relative;
    cursor: pointer;
    animation: indicatorPulse 2s infinite;
}

@keyframes indicatorPulse {
    0%, 100% { transform: scale(1); }
    50% { transform: scale(1.1); }
}

.disease-indicator.copd { background-color: #e74c3c; }
.disease-indicator.fibrosis { background-color: #f39c12; }
.disease-indicator.normal { background-color: #27ae60; }
.disease-indicator.pneumonia { background-color: #e67e22; }
.disease-indicator.tb { background-color: #8e44ad; }

.disease-indicator:hover::after {
    content: attr(data-tooltip);
    position: absolute;
    left: 50%; 
    top: -35px;
    transform: translateX(-50%);
    background: #333;
    color: #fff;
    font-size: 0.82rem;
    padding: 6px 12px;
    border-radius: 8px;
    box-shadow: 0 4px 15px rgba(0,0,0,0.2);
    white-space: nowrap;
    z-index: 10;
    animation: tooltipFloat 0.3s ease;
}

@keyframes tooltipFloat {
    0% { opacity: 0; transform: translateX(-50%) translateY(5px); }
    100% { opacity: 1; transform: translateX(-50%) translateY(0); }
}

.chat-input-area {
    border-top: 1px solid rgba(255,255,255,0.13);
    background: rgba(255,255,255,0.10);
    padding: 18px 16px 16px 16px;
    display: flex;
    flex-direction: column;
    position: relative;
}

.action-buttons {
    display: flex;
    justify-content: space-between;
    gap: 14px;
    margin-bottom: 12px;
}

.action-buttons button {
    flex-grow: 1;
    padding: 13px 0;
    font-size: 1.08rem;
    background: linear-gradient(136deg,#667eea 40%,#764ba2 100%);
    color: white;
    border: none;
    border-radius: 14px;
    cursor: pointer;
    font-weight: 540;
    letter-spacing: 0.02em;
    box-shadow: 0 4px 14px rgba(102, 126, 234, 0.06);
    transition: all 0.3s cubic-bezier(0.4,0,0.2,1);
    outline: none;
    position: relative;
    overflow: hidden;
}

.action-buttons button::before {
    content: '';
    position: absolute;
    top: 0;
    left: -100%;
    width: 100%;
    height: 100%;
    background: linear-gradient(90deg, transparent, rgba(255,255,255,0.3), transparent);
    transition: left 0.5s;
}

.action-buttons button:hover::before {
    left: 100%;
}

.action-buttons button:hover {
    background: linear-gradient(136deg,#667eea 20%,#b191db 100%);
    transform: translateY(-3px) scale(1.05);
    box-shadow: 0 8px 30px rgba(118, 75, 162, 0.2);
}

.action-buttons button:active {
    transform: translateY(-1px) scale(1.02);
}

.action-buttons button:disabled {
    background: linear-gradient(136deg, #b8bdd9 30%, #c7b7e4 70%);
    cursor: not-allowed;
    opacity: 0.6;
    transform: none;
}

.message-input-bar { 
    display: flex; 
    margin-top: 2px;
    position: relative;
}

#message-input {
    width: 100%;
    border: 1px solid rgba(145,145,180,0.26);
    border-radius: 14px;
    padding: 13px;
    font-size: 1.09rem;
    background: rgba(255,255,255,0.18);
    color: #212941;
    transition: all 0.3s ease;
}

#message-input::placeholder { 
    color: rgba(145,145,180,0.56);
}

#message-input:focus { 
    outline: none; 
    border-color: #764ba2; 
    box-shadow: 0 0 15px rgba(118, 75, 162, 0.3);
    background: rgba(255,255,255,0.25);
}

.typing { 
    border-left: 3px solid #764ba2; 
    animation: blink 0.8s steps(1) infinite;
}

@keyframes blink { 
    0%,50%,100%{ border-color:transparent;} 
    25%,75%{ border-color:#764ba2;} 
}

.doctor-card { 
    background: var(--card-bg); 
    padding: 12px; 
    margin: 6px 0; 
    border-radius: 13px;
    transition: transform 0.3s ease, box-shadow 0.3s ease;
    border-left: 3px solid #764ba2;
}

.doctor-card:hover {
    transform: translateX(5px);
    box-shadow: 0 5px 20px rgba(118, 75, 162, 0.1);
}

.doctor-card a { 
    color: #764ba2; 
    text-decoration: none; 
    font-weight: 500; 
}

.doctor-card a:hover { 
    text-decoration: underline; 
}

.pdf-preview { 
    background: var(--card-bg); 
    padding: 13px; 
    margin: 6px 0; 
    border-radius: 13px; 
    text-align: center;
    animation: pulseGlow 2s infinite;
}

@keyframes pulseGlow {
    0%, 100% { box-shadow: 0 0 5px rgba(118, 75, 162, 0.3); }
    50% { box-shadow: 0 0 20px rgba(118, 75, 162, 0.6); }
}

.pdf-preview a { 
    color: #764ba2; 
    font-weight: 600; 
    text-decoration: none; 
}

.pdf-preview a:hover { 
    text-decoration: underline; 
}

/* Loading Animation */
.loading-dots {
    display: inline-block;
    position: relative;
    width: 40px;
    height: 10px;
}

.loading-dots div {
    position: absolute;
    top: 0;
    width: 6px;
    height: 6px;
    border-radius: 50%;
    background: #764ba2;
    animation-timing-function: cubic-bezier(0, 1, 1, 0);
}

.loading-dots div:nth-child(1) {
    left: 4px;
    animation: loading1 0.6s infinite;
}

.loading-dots div:nth-child(2) {
    left: 4px;
    animation: loading2 0.6s infinite;
}

.loading-dots div:nth-child(3) {
    left: 16px;
    animation: loading2 0.6s infinite;
}

.loading-dots div:nth-child(4) {
    left: 28px;
    animation: loading3 0.6s infinite;
}

@keyframes loading1 {
    0% { transform: scale(0); }
    100% { transform: scale(1); }
}

@keyframes loading3 {
    0% { transform: scale(1); }
    100% { transform: scale(0); }
}

@keyframes loading2 {
    0% { transform: translate(0, 0); }
    100% { transform: translate(12px, 0); }
}

/* Responsive */
@media (max-width: 600px) {
    .container { 
        max-width: 95vw; 
        border-radius: 20px; 
        height: 95vh;
    }
    .chat-header { 
        font-size: 1.1rem; 
        padding: 16px;
    }
    .chat-window { 
        padding: 16px; 
    }
    .action-buttons button { 
        font-size: 0.95rem; 
        padding: 11px 0;
    }
    .floating-circle {
        opacity: 0.5;
    }
}

/* Particle Effects */
.particles {
    position: fixed;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    pointer-events: none;
    z-index: 0;
}

.particle {
    position: absolute;
    width: 4px;
    height: 4px;
    background: rgba(255, 255, 255, 0.6);
    border-radius: 50%;
    animation: particleFloat 10s infinite linear;
}

@keyframes particleFloat {
    0% { 
        transform: translateY(100vh) translateX(0px) scale(0);
        opacity: 0;
    }
    10% {
        opacity: 1;
        transform: scale(1);
    }
    90% {
        opacity: 1;
    }
    100% { 
        transform: translateY(-100px) translateX(100px) scale(0);
        opacity: 0;
    }
}

</style>
</head>
<body>
<!-- Floating Background Elements -->
<div class="floating-elements">
    <div class="floating-circle"></div>
    <div class="floating-circle"></div>
    <div class="floating-circle"></div>
    <div class="floating-circle"></div>
    <div class="floating-circle"></div>
</div>

<!-- Floating Medical Icons -->
<div class="medical-icons">
    <div class="medical-icon">⚕</div>
    <div class="medical-icon">🫁</div>
    <div class="medical-icon">💊</div>
    <div class="medical-icon">🩺</div>
</div>

<!-- Particles -->
<div class="particles" id="particles"></div>

<div class="container">
    <div class="chat-header">
        Multi-Chronic Disease Detection Chatbot
        <div class="status-dot"></div>
    </div>
    <div class="chat-window" id="chat-window"></div>
    <div class="chat-input-area">
        <div class="action-buttons">
            <button id="find-doctors-btn" disabled title="Find nearby respiratory doctors">Find Doctors</button>
            <button id="download-report-btn" disabled title="Download medical PDF report">Download Report</button>
            <button id="upload-xray-btn" title="Upload your X-ray image for analysis">Upload X-ray</button>
        </div>
        <div class="message-input-bar">
            <input type="text" id="message-input" placeholder="Type a message..." disabled aria-label="Chat input">
        </div>
    </div>
    <input type="file" id="file-input" accept="image/*" style="display:none;">
</div>

<script>
// Create floating particles
function createParticles() {
    const particlesContainer = document.getElementById('particles');
    
    for (let i = 0; i < 50; i++) {
        const particle = document.createElement('div');
        particle.className = 'particle';
        particle.style.left = Math.random() * 100 + '%';
        particle.style.animationDelay = Math.random() * 10 + 's';
        particle.style.animationDuration = (Math.random() * 10 + 10) + 's';
        particlesContainer.appendChild(particle);
    }
}

// Initialize particles
createParticles();

function getDiseaseIndicator(disease) {
    const indicators = {
        "COPD": {class: "copd", text: "COPD - Chronic Obstructive Pulmonary Disease"},
        "fibrosis": {class: "fibrosis", text: "Fibrosis - Lung Scarring"},
        "normal": {class: "normal", text: "Normal - Healthy Lungs"},
        "pneumonia": {class: "pneumonia", text: "Pneumonia - Lung Infection"},
        "pulmonary tb": {class: "tb", text: "Tuberculosis - Bacterial Infection"}
    };
    return indicators[disease] || {class: "normal", text: "Normal - Healthy Lungs"};
}

const chatWindow = document.getElementById('chat-window');
const uploadBtn = document.getElementById('upload-xray-btn');
const fileInput = document.getElementById('file-input');
const doctorsBtn = document.getElementById('find-doctors-btn');
const reportBtn = document.getElementById('download-report-btn');
const messageInput = document.getElementById('message-input');

function addMessage(message, sender='bot') {
    const div = document.createElement('div');
    div.className = `message ${sender}-message`;
    div.innerHTML = message;
    chatWindow.appendChild(div);
    chatWindow.scrollTop = chatWindow.scrollHeight;
}

function addLoadingMessage() {
    const loadingHtml = `
        Processing your X-ray image... 
        <div class="loading-dots">
            <div></div><div></div><div></div><div></div>
        </div>
    `;
    addMessage(loadingHtml, 'bot');
}

// Welcome message with enhanced styling
addMessage(`
    <strong>Hello! Welcome to our AI Medical Assistant</strong><br>
    <span style='color:#767c8b;font-size:.95em;'>
        Upload your chest X-ray to begin advanced disease detection analysis.
        Our AI system can identify multiple chronic conditions with high accuracy.
    </span>
`, 'bot');

// Upload X-ray with enhanced feedback
uploadBtn.addEventListener('click', ()=>fileInput.click());

fileInput.addEventListener('change', async(event)=>{
    const file = event.target.files[0];
    if(!file) return;
    
    addMessage(`📤 <strong>Uploading:</strong> ${file.name}<br><span style='font-size:0.9em;color:#767c8b;'>File size: ${(file.size/1024/1024).toFixed(2)} MB</span>`, 'bot');
    
    const formData = new FormData();
    formData.append('file',file);

    try {
        addLoadingMessage();
        const response = await fetch('/predict',{method:'POST',body:formData});
        const data = await response.json();
        
        // Remove loading message
        const messages = chatWindow.querySelectorAll('.message');
        const lastMessage = messages[messages.length - 1];
        if (lastMessage.innerHTML.includes('loading-dots')) {
            lastMessage.remove();
        }
        
        if(data.status==='success'){
            const indicatorInfo = getDiseaseIndicator(data.disease);
            addMessage(
                `<span class="disease-indicator ${indicatorInfo.class}" data-tooltip="${indicatorInfo.text}"></span>
                 <strong>Analysis Complete!</strong><br>
                 <strong>Detected Condition:</strong> ${data.disease.toUpperCase()}<br>
                 <span style='color:#767c8b;font-size:.9em;'>
                    Hover over the colored indicator for more information
                 </span>
                 `,'bot');
            doctorsBtn.disabled=false;
            reportBtn.disabled=false;
            messageInput.disabled=false;
        } else {
            addMessage(`❌ <strong>Error:</strong> ${data.error}`, 'bot');
        }
    } catch(err){
        addMessage("⚠️ <strong>Connection Failed:</strong> Please check your internet connection and try again.", 'bot');
    }
});

// Enhanced report download
reportBtn.addEventListener('click', ()=>{
    addMessage(`
        <div class='pdf-preview'>
            📄 <strong>Generating comprehensive medical report...</strong><br>
            <span style='font-size:0.9em;color:#767c8b;'>Including analysis results, recommendations, and disclaimers</span><br>
            <a href='/generate_report' target='_blank'>📥 View & Download PDF Report</a>
        </div>
    `, 'bot');
});

// Enhanced doctor finder
doctorsBtn.addEventListener('click',()=>{
    addMessage("🔍 <strong>Locating nearby medical specialists...</strong><br><span style='font-size:0.9em;color:#767c8b;'>Searching for respiratory and pulmonary experts in your area</span>",'bot');
    
    if(!navigator.geolocation){
        addMessage("❌ <strong>Location Error:</strong> Geolocation is not supported by your browser. Please enable location services.", 'bot');
        return;
    }
    
    navigator.geolocation.getCurrentPosition(async(pos)=>{
        const {latitude,longitude}=pos.coords;
        try{
            const resp = await fetch('/get_doctors',{
                method:'POST',
                headers:{'Content-Type':'application/json'},
                body:JSON.stringify({latitude,longitude})
            });
            const data = await resp.json();
            if(data.doctors && data.doctors.length>0){
                let html="🏥 <strong>Found Nearby Specialists:</strong><br><br>";
                data.doctors.slice(0,5).forEach((doc, index)=>{
                    html+=`<div class='doctor-card'>
                        <strong>${index + 1}. ${doc.name}</strong><br>
                        <a href='https://www.google.com/maps/search/?api=1&query=${encodeURIComponent(doc.location)}' target='_blank'>
                            📍 ${doc.location}
                        </a>
                    </div>`;
                });
                addMessage(html,'bot');
            } else {
                addMessage("❌ <strong>No Results:</strong> No medical specialists found in your immediate area. Try expanding your search radius or consult your local healthcare directory.",'bot');
            }
        }catch(err){
            addMessage("⚠️ <strong>Search Failed:</strong> Unable to fetch doctor information. Please try again later.",'bot');
        }
    }, (error) => {
        let errorMsg = "❌ <strong>Location Access Denied:</strong> ";
        switch(error.code) {
            case error.PERMISSION_DENIED:
                errorMsg += "Please enable location permissions in your browser settings.";
                break;
            case error.POSITION_UNAVAILABLE:
                errorMsg += "Location information is unavailable.";
                break;
            case error.TIMEOUT:
                errorMsg += "Location request timed out.";
                break;
            default:
                errorMsg += "An unknown error occurred.";
                break;
        }
        addMessage(errorMsg, 'bot');
    });
});

// Add some interactive chat functionality
messageInput.addEventListener('keypress', (e) => {
    if (e.key === 'Enter' && !messageInput.disabled) {
        const message = messageInput.value.trim();
        if (message) {
            addMessage(message, 'user');
            messageInput.value = '';
            
            // Simulate bot response
            setTimeout(() => {
                addMessage("Thank you for your message! For detailed medical advice, please consult with a healthcare professional or use our diagnostic tools.", 'bot');
            }, 1000);
        }
    }
});
</script>
</body>
</html>

"""

# -------------------------------
# Utilities
# -------------------------------
def preprocess_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img,(224,224))
    img = img/255.0
    img = np.expand_dims(img,axis=0)
    return img

# -------------------------------
# Routes
# -------------------------------
@app.route("/")
def index(): 
    return render_template_string(chatbot_html)

@app.route("/predict",methods=["POST"])
def predict():
    if not all([cnn_model,rf_model,feature_extractor]):
        return jsonify({"status":"error","error":"Models not loaded."}),500
    if "file" not in request.files: 
        return jsonify({"status":"error","error":"No file part"}),400
    file=request.files["file"]
    if file.filename=='': 
        return jsonify({"status":"error","error":"No selected file"}),400
    try:
        uploads_dir=os.path.join(os.getcwd(),'uploads')
        os.makedirs(uploads_dir,exist_ok=True)
        xray_path=os.path.join(uploads_dir,file.filename)
        file.save(xray_path)
        img=preprocess_image(xray_path)
        features=feature_extractor.predict(img).reshape(1,-1)
        rf_prediction=rf_model.predict(features)[0]
        disease=DISEASE_CLASSES[int(rf_prediction)]
        session['disease']=disease
        session['xray_path']=xray_path
        return jsonify({"status":"success","disease":disease})
    except Exception as e:
        logging.error(f"Prediction error: {e}")
        return jsonify({"status":"error","error":str(e)}),500

@app.route("/get_doctors", methods=["POST"])
def get_doctors():
    data = request.get_json()
    latitude = data.get("latitude")
    longitude = data.get("longitude")
    if latitude is None or longitude is None:
        return jsonify({"error": "Location not provided"}), 400

    url = "https://maps.googleapis.com/maps/api/place/nearbysearch/json"
    params = {
        "location": f"{latitude},{longitude}",
        "radius": 15000,  # 15 km radius
        "type": "doctor",
        "keyword": "pulmonologist",
        "key": GOMAPS_API_KEY
    }

    try:
        response = requests.get(url, params=params, verify=False)
        response.raise_for_status()
        results = response.json().get("results", [])
        doctors = [{"name": r.get("name"), "location": r.get("vicinity", "Address not available")} for r in results]
        return jsonify({"doctors": doctors})
    except requests.exceptions.RequestException as e:
        logging.error(f"Google Maps API request failed: {e}")
        return jsonify({"doctors": [], "error": "Failed to fetch doctors"}), 500

@app.route("/generate_report")
def generate_report():
    disease = session.get('disease')
    xray_path = session.get('xray_path')
    if not disease or not xray_path:
        return "No data available to generate report.", 400

    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, "Multi-Chronic Disease Detection Report", ln=True, align="C")
    pdf.ln(10)

    pdf.set_font("Arial", "", 12)
    pdf.cell(0, 10, f"Disease Detected: {disease.upper()}", ln=True)
    pdf.ln(5)
    
    # Add disease description instead of emoji
    disease_descriptions = {
        "COPD": "Chronic Obstructive Pulmonary Disease",
        "fibrosis": "Pulmonary Fibrosis",
        "normal": "Normal/Healthy Lungs",
        "pneumonia": "Pneumonia Infection",
        "pulmonary tb": "Pulmonary Tuberculosis"
    }
    
    description = disease_descriptions.get(disease, "Unknown condition")
    pdf.cell(0, 10, f"Description: {description}", ln=True)
    pdf.ln(10)

    # Insert X-ray image
    if os.path.exists(xray_path):
        pdf.image(xray_path, x=30, w=150)
        pdf.ln(100)  # Add space after image
    
    # Add recommendations
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 10, "Recommendations:", ln=True)
    pdf.ln(5)
    
    pdf.set_font("Arial", "", 10)
    recommendations = {
        "normal": "Continue maintaining healthy lifestyle. Regular check-ups recommended.",
        "COPD": "Immediate consultation with pulmonologist required. Avoid smoking.",
        "fibrosis": "Urgent medical attention needed. Consult respiratory specialist.",
        "pneumonia": "Seek immediate medical treatment. Complete prescribed antibiotics.",
        "pulmonary tb": "Immediate isolation and medical care required. Follow treatment protocol."
    }
    
    recommendation = recommendations.get(disease, "Consult healthcare professional for proper diagnosis.")
    pdf.multi_cell(0, 5, recommendation)
    pdf.ln(10)
    
    # Add disclaimer
    pdf.set_font("Arial", "I", 9)
    disclaimer = "This AI analysis is for informational purposes only. Please consult qualified healthcare professionals for proper medical diagnosis and treatment."
    pdf.multi_cell(0, 5, disclaimer)

    # Save PDF temporarily
    reports_dir = os.path.join(os.getcwd(), 'reports')
    os.makedirs(reports_dir, exist_ok=True)
    pdf_path = os.path.join(reports_dir, f"medical_report_{disease.replace(' ', '_')}.pdf")
    pdf.output(pdf_path)

    return send_file(pdf_path, as_attachment=True, download_name=f"medical_report_{disease.replace(' ', '_')}.pdf")

# -------------------------------
# Run App
# -------------------------------
if __name__ == "__main__":
    app.run(debug=True)
