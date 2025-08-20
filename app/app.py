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
from werkzeug.utils import secure_filename
import traceback
from datetime import datetime
import threading
import time
import atexit
import gc
import resource

# Disable SSL warnings
from requests.packages.urllib3.exceptions import InsecureRequestWarning
requests.packages.urllib3.disable_warnings(InsecureRequestWarning)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.secret_key = 'your_super_secret_key_change_in_production'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
CORS(app)

# -------------------------------
# CONFIGURATION
# -------------------------------
GOMAPS_API_KEY = "YOUR_GOMAPS_API_KEY"  # Replace with your actual API key
DISEASE_CLASSES = ["COPD", "fibrosis", "normal", "pneumonia", "pulmonary tb"]
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'tiff'}

# Global variables for models
cnn_model = None
rf_model = None
feature_extractor = None
models_loaded = False
startup_complete = False

# -------------------------------
# MEMORY MONITORING (WITHOUT PSUTIL)
# -------------------------------
def get_memory_usage():
    """Get current memory usage in MB using resource module"""
    try:
        # Get memory usage in bytes and convert to MB
        usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        # On Linux, ru_maxrss is in KB, on macOS/BSD it's in bytes
        if os.name == 'posix':
            return usage / 1024  # Convert KB to MB on Linux
        else:
            return usage / (1024 * 1024)  # Convert bytes to MB on other systems
    except:
        return 0

# -------------------------------
# MODEL LOADING WITH MEMORY OPTIMIZATION
# -------------------------------
def load_models():
    """Load models with memory optimization and proper error handling"""
    global cnn_model, rf_model, feature_extractor, models_loaded
    
    try:
        logger.info(f"🔄 Starting model loading process... Current memory: {get_memory_usage():.2f}MB")
        
        # Base directory setup
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        ROOT_DIR = os.path.abspath(os.path.join(BASE_DIR, ".."))
        
        # Model paths
        cnn_model_path = os.path.join(ROOT_DIR, "models", "densenet_new_finetuned_v3.h5")
        rf_model_path = os.path.join(ROOT_DIR, "models", "fast_rf_xgb_stack2.pkl")
        
        # Check if model files exist
        if not os.path.exists(cnn_model_path):
            logger.error(f"❌ CNN model not found at: {cnn_model_path}")
            return False
            
        if not os.path.exists(rf_model_path):
            logger.error(f"❌ RF model not found at: {rf_model_path}")
            return False
        
        # Configure TensorFlow for memory efficiency
        try:
            # Limit TensorFlow memory growth
            physical_devices = tf.config.list_physical_devices('GPU')
            if physical_devices:
                tf.config.experimental.set_memory_growth(physical_devices[0], True)
            else:
                tf.config.set_visible_devices([], 'GPU')
        except Exception as e:
            logger.warning(f"TensorFlow GPU config warning: {e}")
        
        # Reduce thread usage to save memory
        tf.config.threading.set_intra_op_parallelism_threads(1)
        tf.config.threading.set_inter_op_parallelism_threads(1)
        
        # Load CNN model with memory optimization
        logger.info("🔄 Loading CNN model...")
        cnn_model = tf.keras.models.load_model(cnn_model_path, compile=False)
        gc.collect()  # Force garbage collection
        logger.info(f"✅ CNN model loaded successfully. Memory: {get_memory_usage():.2f}MB")
        
        # Load Random Forest model
        logger.info("🔄 Loading Random Forest model...")
        rf_model = joblib.load(rf_model_path)
        gc.collect()
        logger.info(f"✅ Random Forest model loaded successfully. Memory: {get_memory_usage():.2f}MB")
        
        # Create feature extractor
        logger.info("🔄 Creating feature extractor...")
        try:
            feature_extractor = Model(
                inputs=cnn_model.input,
                outputs=cnn_model.get_layer("global_average_pooling2d").output
            )
            logger.info("✅ Feature extractor created successfully")
        except Exception as e:
            logger.warning(f"⚠️ Could not find 'global_average_pooling2d' layer, trying alternatives...")
            # Try to find a suitable layer for feature extraction
            for layer in reversed(cnn_model.layers):
                if 'pool' in layer.name.lower() or 'flatten' in layer.name.lower():
                    feature_extractor = Model(inputs=cnn_model.input, outputs=layer.output)
                    logger.info(f"✅ Using layer '{layer.name}' for feature extraction")
                    break
            else:
                logger.error("❌ Could not create feature extractor")
                return False
        
        gc.collect()
        models_loaded = True
        logger.info(f"🎉 All models loaded successfully! Final memory: {get_memory_usage():.2f}MB")
        return True
        
    except Exception as e:
        logger.error(f"❌ Error loading models: {str(e)}")
        logger.error(traceback.format_exc())
        return False

def allowed_file(filename):
    """Check if file has allowed extension"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# -------------------------------
# UTILITIES
# -------------------------------
def preprocess_image(image_path):
    """Preprocess image with memory optimization"""
    try:
        logger.info(f"🔄 Preprocessing image: {image_path}")
        
        # Read image
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not read image from {image_path}")
        
        # Resize image to reduce memory usage
        img = cv2.resize(img, (224, 224))
        
        # Normalize and convert to float32 (uses less memory than float64)
        img = img.astype(np.float32) / 255.0
        
        # Add batch dimension
        img = np.expand_dims(img, axis=0)
        
        logger.info("✅ Image preprocessed successfully")
        return img
        
    except Exception as e:
        logger.error(f"❌ Error preprocessing image: {str(e)}")
        raise

def cleanup_old_files():
    """Clean up old uploaded files and reports"""
    try:
        uploads_dir = os.path.join(os.getcwd(), 'uploads')
        reports_dir = os.path.join(os.getcwd(), 'reports')
        
        current_time = time.time()
        
        # Clean uploads older than 1 hour
        if os.path.exists(uploads_dir):
            for filename in os.listdir(uploads_dir):
                file_path = os.path.join(uploads_dir, filename)
                if os.path.isfile(file_path):
                    if current_time - os.path.getmtime(file_path) > 3600:  # 1 hour
                        os.remove(file_path)
                        logger.info(f"🗑️ Cleaned up old upload: {filename}")
        
        # Clean reports older than 1 hour
        if os.path.exists(reports_dir):
            for filename in os.listdir(reports_dir):
                file_path = os.path.join(reports_dir, filename)
                if os.path.isfile(file_path):
                    if current_time - os.path.getmtime(file_path) > 3600:  # 1 hour
                        os.remove(file_path)
                        logger.info(f"🗑️ Cleaned up old report: {filename}")
        
        # Force garbage collection after cleanup
        gc.collect()
                        
    except Exception as e:
        logger.warning(f"⚠️ Error during cleanup: {str(e)}")

def ensure_startup():
    """Ensure startup tasks are completed"""
    global startup_complete
    
    if not startup_complete:
        logger.info("🚀 Starting Multi-Chronic Disease Detection System...")
        
        # Load models in background thread
        def load_models_background():
            success = load_models()
            if success:
                logger.info("🎉 System ready for predictions!")
            else:
                logger.error("❌ System startup failed - models not loaded")
        
        # Start model loading in background
        threading.Thread(target=load_models_background, daemon=True).start()
        
        # Start cleanup scheduler
        def periodic_cleanup():
            while True:
                time.sleep(3600)  # Run every hour
                cleanup_old_files()
        
        threading.Thread(target=periodic_cleanup, daemon=True).start()
        startup_complete = True

# -------------------------------
# HTML TEMPLATE
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
    --error-color: #e74c3c;
    --success-color: #27ae60;
    --warning-color: #f39c12;
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

@keyframes float {
    0%, 100% { transform: translateY(0px) rotate(0deg); }
    25% { transform: translateY(-20px) rotate(90deg); }
    50% { transform: translateY(-40px) rotate(180deg); }
    75% { transform: translateY(-20px) rotate(270deg); }
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

.status-dot {
    position: absolute;
    top: 20px;
    right: 20px;
    width: 12px;
    height: 12px;
    background: var(--success-color);
    border-radius: 50%;
    box-shadow: 0 0 0 4px rgba(39, 174, 96, 0.3);
    animation: pulse 2s infinite;
}

.status-dot.error {
    background: var(--error-color);
    box-shadow: 0 0 0 4px rgba(231, 76, 60, 0.3);
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

.error-message {
    background: linear-gradient(135deg, #e74c3c, #c0392b);
    color: #fff;
    border-left: 3px solid rgba(255, 255, 255, 0.5);
}

.success-message {
    background: linear-gradient(135deg, #27ae60, #2ecc71);
    color: #fff;
    border-left: 3px solid rgba(255, 255, 255, 0.5);
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

.action-buttons button:hover:not(:disabled) {
    background: linear-gradient(136deg,#667eea 20%,#b191db 100%);
    transform: translateY(-3px) scale(1.05);
    box-shadow: 0 8px 30px rgba(118, 75, 162, 0.2);
}

.action-buttons button:disabled {
    background: linear-gradient(136deg, #b8bdd9 30%, #c7b7e4 70%);
    cursor: not-allowed;
    opacity: 0.6;
    transform: none;
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
}
</style>
</head>
<body>
<div class="floating-elements">
    <div class="floating-circle"></div>
    <div class="floating-circle"></div>
    <div class="floating-circle"></div>
</div>

<div class="container">
    <div class="chat-header">
        Multi-Chronic Disease Detection Chatbot
        <div class="status-dot" id="status-dot"></div>
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
const statusDot = document.getElementById('status-dot');

function addMessage(message, sender='bot', type='normal') {
    const div = document.createElement('div');
    let className = `message ${sender}-message`;
    if (type === 'error') className += ' error-message';
    if (type === 'success') className += ' success-message';
    
    div.className = className;
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

function removeLastMessage() {
    const messages = chatWindow.querySelectorAll('.message');
    if (messages.length > 0) {
        const lastMessage = messages[messages.length - 1];
        if (lastMessage.innerHTML.includes('loading-dots')) {
            lastMessage.remove();
        }
    }
}

function updateStatusDot(isHealthy) {
    if (isHealthy) {
        statusDot.classList.remove('error');
    } else {
        statusDot.classList.add('error');
    }
}

async function checkHealth() {
    try {
        const response = await fetch('/health');
        const data = await response.json();
        updateStatusDot(data.status === 'healthy' && data.models_loaded);
        
        if (!data.models_loaded) {
            addMessage("⚠️ <strong>System Warning:</strong> AI models are loading. Please wait a moment before uploading images.", 'bot', 'error');
        }
    } catch (error) {
        updateStatusDot(false);
        addMessage("❌ <strong>System Error:</strong> Cannot connect to server. Please refresh the page.", 'bot', 'error');
    }
}

checkHealth();
setInterval(checkHealth, 10000); // Check every 10 seconds

addMessage(`
    <strong>Hello! Welcome to our AI Medical Assistant</strong><br>
    <span style='color:#767c8b;font-size:.95em;'>
        Upload your chest X-ray to begin advanced disease detection analysis.
        Our AI system can identify multiple chronic conditions with high accuracy.
    </span>
`, 'bot');

uploadBtn.addEventListener('click', () => fileInput.click());

fileInput.addEventListener('change', async(event) => {
    const file = event.target.files[0];
    if (!file) return;
    
    const validTypes = ['image/jpeg', 'image/png', 'image/gif', 'image/bmp', 'image/tiff'];
    if (!validTypes.includes(file.type)) {
        addMessage("❌ <strong>Invalid File Type:</strong> Please upload a valid image file (JPEG, PNG, GIF, BMP, or TIFF).", 'bot', 'error');
        return;
    }
    
    if (file.size > 16 * 1024 * 1024) {
        addMessage("❌ <strong>File Too Large:</strong> Please upload an image smaller than 16MB.", 'bot', 'error');
        return;
    }
    
    addMessage(`📤 <strong>Uploading:</strong> ${file.name}<br><span style='font-size:0.9em;color:#767c8b;'>File size: ${(file.size/1024/1024).toFixed(2)} MB</span>`, 'bot');
    
    const formData = new FormData();
    formData.append('file', file);

    try {
        addLoadingMessage();
        uploadBtn.disabled = true;
        uploadBtn.textContent = 'Processing...';
        
        const controller = new AbortController();
        const timeoutId = setTimeout(() => controller.abort(), 180000); // 3 minutes timeout
        
        const response = await fetch('/predict', {
            method: 'POST',
            body: formData,
            signal: controller.signal
        });
        
        clearTimeout(timeoutId);
        const data = await response.json();
        
        removeLastMessage();
        
        if (data.status === 'success') {
            const indicatorInfo = getDiseaseIndicator(data.disease);
            addMessage(
                `<span class="disease-indicator ${indicatorInfo.class}" data-tooltip="${indicatorInfo.text}"></span>
                 <strong>Analysis Complete!</strong><br>
                 <strong>Detected Condition:</strong> ${data.disease.toUpperCase()}<br>
                 <span style='color:#767c8b;font-size:.9em;'>
                    Hover over the colored indicator for more information
                 </span>
                `, 'bot', 'success');
            doctorsBtn.disabled = false;
            reportBtn.disabled = false;
            messageInput.disabled = false;
        } else {
            addMessage(`❌ <strong>Error:</strong> ${data.error}`, 'bot', 'error');
        }
    } catch (err) {
        removeLastMessage();
        if (err.name === 'AbortError') {
            addMessage("⏱️ <strong>Timeout:</strong> Processing took too long. Server may be under heavy load. Please try again later.", 'bot', 'error');
        } else {
            addMessage("⚠️ <strong>Connection Failed:</strong> Server resources may be exhausted. Please try again in a few moments.", 'bot', 'error');
        }
    } finally {
        uploadBtn.disabled = false;
        uploadBtn.textContent = 'Upload X-ray';
    }
});

reportBtn.addEventListener('click', () => {
    addMessage(`
        <div class='pdf-preview'>
            📄 <strong>Generating comprehensive medical report...</strong><br>
            <span style='font-size:0.9em;color:#767c8b;'>Including analysis results, recommendations, and disclaimers</span><br>
            <a href='/generate_report' target='_blank'>📥 View & Download PDF Report</a>
        </div>
    `, 'bot');
});

doctorsBtn.addEventListener('click', () => {
    addMessage("🔍 <strong>Locating nearby medical specialists...</strong><br><span style='font-size:0.9em;color:#767c8b;'>Searching for respiratory and pulmonary experts in your area</span>", 'bot');
    
    if (!navigator.geolocation) {
        addMessage("❌ <strong>Location Error:</strong> Geolocation is not supported by your browser. Please enable location services.", 'bot', 'error');
        return;
    }
    
    navigator.geolocation.getCurrentPosition(async(pos) => {
        const {latitude, longitude} = pos.coords;
        try {
            const resp = await fetch('/get_doctors', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({latitude, longitude})
            });
            const data = await resp.json();
            if (data.doctors && data.doctors.length > 0) {
                let html = "🏥 <strong>Found Nearby Specialists:</strong><br><br>";
                data.doctors.slice(0, 5).forEach((doc, index) => {
                    html += `<div class='doctor-card'>
                        <strong>${index + 1}. ${doc.name}</strong><br>
                        <a href='https://www.google.com/maps/search/?api=1&query=${encodeURIComponent(doc.location)}' target='_blank'>
                            📍 ${doc.location}
                        </a>
                    </div>`;
                });
                addMessage(html, 'bot');
            } else {
                addMessage("❌ <strong>No Results:</strong> No medical specialists found in your immediate area.", 'bot', 'error');
            }
        } catch (err) {
            addMessage("⚠️ <strong>Search Failed:</strong> Unable to fetch doctor information. Please try again later.", 'bot', 'error');
        }
    }, (error) => {
        addMessage("❌ <strong>Location Access Denied:</strong> Please enable location permissions.", 'bot', 'error');
    });
});

messageInput.addEventListener('keypress', (e) => {
    if (e.key === 'Enter' && !messageInput.disabled) {
        const message = messageInput.value.trim();
        if (message) {
            addMessage(message, 'user');
            messageInput.value = '';
            
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
# ROUTES
# -------------------------------
@app.route("/")
def index(): 
    ensure_startup()  # Ensure startup tasks run on first request
    return render_template_string(chatbot_html)

@app.route("/health")
def health_check():
    """Health check endpoint"""
    ensure_startup()  # Ensure startup tasks run
    return jsonify({
        "status": "healthy",
        "models_loaded": models_loaded,
        "memory_usage_mb": get_memory_usage(),
        "timestamp": datetime.now().isoformat()
    })

@app.route("/predict", methods=["POST"])
def predict():
    """Memory-optimized prediction endpoint"""
    ensure_startup()  # Ensure startup tasks run
    
    logger.info(f"🔄 Starting prediction. Memory before: {get_memory_usage():.2f}MB")
    
    if not models_loaded:
        logger.error("Models not loaded")
        return jsonify({"status": "error", "error": "AI models are still loading. Please wait a moment and try again."}), 503
    
    if "file" not in request.files: 
        return jsonify({"status": "error", "error": "No file uploaded"}), 400
    
    file = request.files["file"]
    if file.filename == '': 
        return jsonify({"status": "error", "error": "No file selected"}), 400
    
    if not allowed_file(file.filename):
        return jsonify({"status": "error", "error": "Invalid file type. Please upload an image file."}), 400
    
    try:
        # Create uploads directory
        uploads_dir = os.path.join(os.getcwd(), 'uploads')
        os.makedirs(uploads_dir, exist_ok=True)
        
        # Secure filename
        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_")
        filename = timestamp + filename
        xray_path = os.path.join(uploads_dir, filename)
        
        logger.info(f"🔄 Saving uploaded file: {filename}")
        file.save(xray_path)
        
        # Preprocess image with memory optimization
        logger.info("🔄 Starting image preprocessing...")
        img = preprocess_image(xray_path)
        logger.info(f"🔄 Image preprocessed. Memory: {get_memory_usage():.2f}MB")
        
        # Extract features with explicit memory management
        logger.info("🔄 Extracting features...")
        
        # Force CPU processing and use batch size of 1
        with tf.device('/CPU:0'):
            features = feature_extractor.predict(img, batch_size=1, verbose=0)
            features = features.reshape(1, -1)
        
        # Clear image from memory immediately
        del img
        gc.collect()
        
        logger.info(f"🔄 Features extracted. Memory: {get_memory_usage():.2f}MB")
        
        # Make prediction
        logger.info("🔄 Making prediction...")
        rf_prediction = rf_model.predict(features)[0]
        disease = DISEASE_CLASSES[int(rf_prediction)]
        
        # Clear features from memory
        del features
        gc.collect()
        
        # Store in session
        session['disease'] = disease
        session['xray_path'] = xray_path
        session['prediction_time'] = datetime.now().isoformat()
        
        logger.info(f"✅ Prediction successful: {disease}. Final memory: {get_memory_usage():.2f}MB")
        
        # Cleanup old files in background
        threading.Thread(target=cleanup_old_files, daemon=True).start()
        
        return jsonify({
            "status": "success",
            "disease": disease,
            "confidence": "high",
            "timestamp": session['prediction_time'],
            "memory_usage_mb": get_memory_usage()
        })
        
    except Exception as e:
        logger.error(f"❌ Prediction error: {str(e)}")
        logger.error(traceback.format_exc())
        
        # Force cleanup on error
        gc.collect()
        
        error_msg = "An error occurred during analysis. Please try again."
        if "memory" in str(e).lower() or "resource" in str(e).lower():
            error_msg = "Server resources are currently limited. Please try again in a few moments or use a smaller image."
        elif "timeout" in str(e).lower():
            error_msg = "Processing timed out. Please try with a smaller image or try again later."
        elif "image" in str(e).lower():
            error_msg = "Invalid image format. Please upload a clear X-ray image."
        
        return jsonify({"status": "error", "error": error_msg}), 500

@app.route("/get_doctors", methods=["POST"])
def get_doctors():
    """Enhanced doctor finder with better error handling"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No location data provided"}), 400
            
        latitude = data.get("latitude")
        longitude = data.get("longitude")
        
        if latitude is None or longitude is None:
            return jsonify({"error": "Invalid location coordinates"}), 400

        # Check if API key is configured
        if GOMAPS_API_KEY == "YOUR_GOMAPS_API_KEY":
            logger.warning("Google Maps API key not configured")
            # Return mock data for demo purposes
            mock_doctors = [
                {"name": "Dr. Smith Pulmonology Clinic", "location": "123 Medical Center Dr, City"},
                {"name": "Respiratory Care Associates", "location": "456 Health Plaza, City"},
                {"name": "City General Hospital - Pulmonology", "location": "789 Hospital Ave, City"}
            ]
            return jsonify({"doctors": mock_doctors})

        url = "https://maps.googleapis.com/maps/api/place/nearbysearch/json"
        params = {
            "location": f"{latitude},{longitude}",
            "radius": 15000,
            "type": "doctor",
            "keyword": "pulmonologist respiratory",
            "key": GOMAPS_API_KEY
        }

        logger.info(f"🔄 Searching for doctors near {latitude}, {longitude}")
        response = requests.get(url, params=params, verify=False, timeout=10)
        response.raise_for_status()
        
        results = response.json().get("results", [])
        doctors = []
        
        for r in results:
            doctor_info = {
                "name": r.get("name", "Unknown Doctor"),
                "location": r.get("vicinity", "Address not available"),
                "rating": r.get("rating", "N/A"),
                "place_id": r.get("place_id", "")
            }
            doctors.append(doctor_info)
        
        logger.info(f"✅ Found {len(doctors)} doctors")
        return jsonify({"doctors": doctors})
        
    except Exception as e:
        logger.error(f"Error in get_doctors: {e}")
        return jsonify({"doctors": [], "error": "Unable to search for doctors at this time."}), 503

@app.route("/generate_report")
def generate_report():
    """Enhanced report generation with better formatting"""
    disease = session.get('disease')
    xray_path = session.get('xray_path')
    prediction_time = session.get('prediction_time')
    
    if not disease:
        return "No diagnosis data available. Please upload an X-ray first.", 400

    try:
        # Create reports directory
        reports_dir = os.path.join(os.getcwd(), 'reports')
        os.makedirs(reports_dir, exist_ok=True)
        
        pdf = FPDF()
        pdf.add_page()
        
        # Header
        pdf.set_font("Arial", "B", 18)
        pdf.cell(0, 15, "Medical AI Analysis Report", ln=True, align="C")
        pdf.ln(5)
        
        # Separator line
        pdf.line(20, pdf.get_y(), 190, pdf.get_y())
        pdf.ln(10)
        
        # Analysis Summary
        pdf.set_font("Arial", "B", 14)
        pdf.cell(0, 10, "Analysis Summary", ln=True)
        pdf.ln(5)
        
        pdf.set_font("Arial", "", 12)
        pdf.cell(0, 8, f"Analysis Date: {datetime.now().strftime('%B %d, %Y at %I:%M %p')}", ln=True)
        if prediction_time:
            pdf.cell(0, 8, f"Processing Time: {prediction_time[:16].replace('T', ' ')}", ln=True)
        pdf.ln(5)
        
        # Diagnosis Section
        pdf.set_font("Arial", "B", 14)
        pdf.cell(0, 10, "AI Diagnosis", ln=True)
        pdf.ln(3)
        
        disease_descriptions = {
            "COPD": ("Chronic Obstructive Pulmonary Disease", "A progressive lung disease that makes breathing difficult."),
            "fibrosis": ("Pulmonary Fibrosis", "Scarring of lung tissue that affects breathing."),
            "normal": ("Normal/Healthy Lungs", "No significant abnormalities detected."),
            "pneumonia": ("Pneumonia", "Infection that inflames air sacs in lungs."),
            "pulmonary tb": ("Pulmonary Tuberculosis", "Bacterial infection primarily affecting the lungs.")
        }
        
        title, description = disease_descriptions.get(disease, ("Unknown Condition", "Further evaluation needed."))
        
        pdf.set_font("Arial", "B", 12)
        pdf.cell(0, 8, f"Detected Condition: {title.upper()}", ln=True)
        pdf.ln(3)
        
        pdf.set_font("Arial", "", 11)
        pdf.multi_cell(0, 6, f"Description: {description}")
        pdf.ln(8)
        
        # Insert X-ray image if available
        if xray_path and os.path.exists(xray_path):
            try:
                pdf.set_font("Arial", "B", 12)
                pdf.cell(0, 8, "Analyzed X-ray Image:", ln=True)
                pdf.ln(5)
                pdf.image(xray_path, x=30, w=150, h=100)
                pdf.ln(105)
            except Exception as e:
                logger.warning(f"Could not insert image in PDF: {e}")
                pdf.cell(0, 8, "X-ray image could not be included in report.", ln=True)
                pdf.ln(10)
        
        # Recommendations Section
        pdf.set_font("Arial", "B", 14)
        pdf.cell(0, 10, "Recommendations", ln=True)
        pdf.ln(3)
        
        pdf.set_font("Arial", "", 11)
        recommendations = {
            "normal": [
                "Continue maintaining a healthy lifestyle",
                "Regular exercise and balanced diet recommended",
                "Schedule routine medical check-ups",
                "Avoid smoking and exposure to pollutants"
            ],
            "COPD": [
                "URGENT: Consult a pulmonologist immediately",
                "Avoid smoking and secondhand smoke",
                "Consider pulmonary rehabilitation programs",
                "Monitor symptoms and seek emergency care if breathing worsens"
            ],
            "fibrosis": [
                "URGENT: Seek immediate medical attention",
                "Consult with a respiratory specialist",
                "Discuss treatment options including medications",
                "Consider joining support groups"
            ],
            "pneumonia": [
                "URGENT: Seek immediate medical treatment",
                "Complete full course of prescribed antibiotics",
                "Rest and stay hydrated",
                "Monitor symptoms and return if condition worsens"
            ],
            "pulmonary tb": [
                "CRITICAL: Immediate isolation and medical care required",
                "Follow strict medication protocol",
                "Notify close contacts for screening",
                "Regular follow-up appointments essential"
            ]
        }
        
        disease_recommendations = recommendations.get(disease, ["Consult healthcare professional for proper evaluation"])
        
        for i, rec in enumerate(disease_recommendations, 1):
            pdf.cell(0, 6, f"{i}. {rec}", ln=True)
        pdf.ln(10)
        
        # Disclaimer Section
        pdf.set_font("Arial", "B", 12)
        pdf.cell(0, 8, "Important Disclaimer", ln=True)
        pdf.ln(3)
        
        pdf.set_font("Arial", "I", 10)
        disclaimer_text = (
            "This AI analysis is provided for informational purposes only and should not be considered "
            "as a substitute for professional medical diagnosis, treatment, or advice. Always consult "
            "with qualified healthcare professionals for proper medical evaluation and treatment decisions. "
            "This system has limitations and may not detect all conditions or may produce false results. "
            "Emergency situations require immediate medical attention regardless of AI analysis results."
        )
        pdf.multi_cell(0, 5, disclaimer_text)
        pdf.ln(10)
        
        # Footer
        pdf.set_font("Arial", "", 8)
        pdf.cell(0, 5, f"Generated by Multi-Chronic Disease Detection System - {datetime.now().strftime('%Y')}", ln=True, align="C")
        
        # Save PDF
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        pdf_filename = f"medical_report_{disease.replace(' ', '_')}_{timestamp}.pdf"
        pdf_path = os.path.join(reports_dir, pdf_filename)
        pdf.output(pdf_path)
        
        logger.info(f"✅ Report generated: {pdf_filename}")
        
        return send_file(
            pdf_path, 
            as_attachment=True, 
            download_name=pdf_filename,
            mimetype='application/pdf'
        )
        
    except Exception as e:
        logger.error(f"❌ Error generating report: {str(e)}")
        return "Error generating report. Please try again.", 500

# -------------------------------
# ERROR HANDLERS
# -------------------------------
@app.errorhandler(413)
def too_large(e):
    return jsonify({"status": "error", "error": "File too large. Maximum size is 16MB."}), 413

@app.errorhandler(404)
def not_found(e):
    return jsonify({"status": "error", "error": "Endpoint not found."}), 404

@app.errorhandler(500)
def server_error(e):
    logger.error(f"Server error: {str(e)}")
    return jsonify({"status": "error", "error": "Internal server error. Please try again."}), 500

# -------------------------------
# CLEANUP ON EXIT
# -------------------------------
def cleanup_on_exit():
    """Cleanup function to run on application exit"""
    logger.info("🧹 Cleaning up resources...")
    cleanup_old_files()

atexit.register(cleanup_on_exit)

# -------------------------------
# RUN APPLICATION
# -------------------------------
if __name__ == "__main__":
    # Load models at startup for development
    load_models()
    
    port = int(os.environ.get("PORT", 5000))
    debug_mode = os.environ.get("FLASK_ENV", "production") == "development"
    
    logger.info(f"🌐 Starting server on port {port}")
    logger.info(f"🔧 Debug mode: {debug_mode}")
    
    app.run(
        host="0.0.0.0", 
        port=port, 
        debug=debug_mode,
        threaded=True
    )
