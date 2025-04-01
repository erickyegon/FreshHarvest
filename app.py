"""
Flask web application for the FreshHarvest project.
Provides a simple web interface for uploading images and getting predictions.
"""

import os
import io
import json
import base64
import cv2
import numpy as np
from pathlib import Path
from datetime import datetime
from PIL import Image
from flask import Flask, request, jsonify, render_template, redirect, url_for

from config import DEVICE, OUTPUT_DIR
from inference import FruitFreshnessInference

# Configuration
MODEL_PATH = "model_outputs/final_model.pt"  # Update this to your model path
UPLOAD_FOLDER = OUTPUT_DIR / "uploads"
RESULTS_FOLDER = OUTPUT_DIR / "results"
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp'}

# Create upload and results folders
UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)
RESULTS_FOLDER.mkdir(parents=True, exist_ok=True)

# Initialize Flask app
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload

# Initialize inference model
try:
    inference = FruitFreshnessInference(MODEL_PATH, DEVICE)
    model_loaded = True
except Exception as e:
    print(f"Error loading model: {e}")
    model_loaded = False
    inference = None

def allowed_file(filename):
    """Check if file has an allowed extension"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    """Render index page"""
    return render_template('index.html', model_loaded=model_loaded)

@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle file upload and make prediction"""
    # Check if model is loaded
    if not model_loaded:
        return jsonify({'error': 'Model not loaded'}), 500
    
    # Check if file is included in the request
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    
    # Check if file is selected
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    # Check if file is allowed
    if not allowed_file(file.filename):
        return jsonify({'error': f'File type not allowed. Use {", ".join(ALLOWED_EXTENSIONS)}'}), 400
    
    try:
        # Save file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{timestamp}_{file.filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Make prediction
        result = inference.predict(filepath, return_probs=True)
        
        # Create annotated image
        annotated_img = inference.create_annotated_image(filepath, result)
        
        # Save annotated image
        annotated_path = os.path.join(RESULTS_FOLDER, f"annotated_{filename}")
        cv2.imwrite(annotated_path, annotated_img)
        
        # Convert annotated image to base64 for display
        _, buffer = cv2.imencode('.jpg', annotated_img)
        img_str = base64.b64encode(buffer).decode('utf-8')
        
        # Get image dimensions for display
        img = Image.open(filepath)
        width, height = img.size
        
        # Prepare response
        response = {
            'prediction': result['prediction'],
            'confidence': result['confidence'],
            'inference_time': result['inference_time'],
            'probabilities': result['probabilities'],
            'image_base64': img_str,
            'filename': filename,
            'width': width,
            'height': height
        }
        
        return jsonify(response)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/predict', methods=['POST'])
def api_predict():
    """API endpoint for prediction"""
    # Check if model is loaded
    if not model_loaded:
        return jsonify({'error': 'Model not loaded'}), 500
    
    # Check for file in request
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    
    # Check if file is selected
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    # Check if file is allowed
    if not allowed_file(file.filename):
        return jsonify({'error': f'File type not allowed. Use {", ".join(ALLOWED_EXTENSIONS)}'}), 400
    
    try:
        # Read image
        img = Image.open(file.stream).convert('RGB')
        
        # Make prediction
        result = inference.predict(img, return_probs=True)
        
        # Prepare API response
        response = {
            'status': 'success',
            'prediction': result['prediction'],
            'confidence': result['confidence'],
            'inference_time': result['inference_time'],
            'probabilities': result['probabilities']
        }
        
        return jsonify(response)
    
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/about')
def about():
    """Render about page"""
    return render_template('about.html')

@app.errorhandler(404)
def page_not_found(e):
    """Handle 404 errors"""
    return render_template('404.html'), 404

@app.errorhandler(500)
def server_error(e):
    """Handle 500 errors"""
    return render_template('500.html'), 500

# Minimal templates for testing - in a real application, create these in a templates folder
@app.route('/generate_templates')
def generate_templates():
    """Generate minimal templates for testing"""
    templates_dir = Path('templates')
    templates_dir.mkdir(exist_ok=True)
    
    # Index template
    index_html = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>FreshHarvest - Fruit Freshness Detection</title>
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <style>
            body { font-family: Arial, sans-serif; margin: 0; padding: 20px; }
            .container { max-width: 800px; margin: 0 auto; }
            h1 { color: #2C8C3C; }
            .upload-area { border: 2px dashed #ccc; padding: 20px; text-align: center; margin: 20px 0; }
            .upload-area:hover { border-color: #2C8C3C; }
            .btn { background-color: #2C8C3C; color: white; padding: 10px 20px; border: none; cursor: pointer; }
            .btn:hover { background-color: #236C2E; }
            .result { margin-top: 20px; display: none; }
            img { max-width: 100%; }
            .error { color: red; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>FreshHarvest Fruit Freshness Detection</h1>
            
            {% if not model_loaded %}
            <div class="error">
                <p>Error: Model not loaded. Please check the model path.</p>
            </div>
            {% else %}
            <p>Upload an image of a fruit to detect its freshness.</p>
            
            <div class="upload-area" id="drop-area">
                <form id="upload-form" enctype="multipart/form-data">
                    <input type="file" id="file-input" name="file" accept=".jpg,.jpeg,.png,.bmp" style="display: none;">
                    <label for="file-input" class="btn">Select Image</label>
                    <p>Or drag and drop an image here</p>
                </form>
            </div>
            
            <div id="result" class="result">
                <h2>Prediction: <span id="prediction"></span></h2>
                <p>Confidence: <span id="confidence"></span></p>
                <p>Inference Time: <span id="inference-time"></span> seconds</p>
                
                <div id="image-container">
                    <img id="result-image" src="" alt="Annotated Image">
                </div>
            </div>
            
            <script>
                // Handle file upload
                const dropArea = document.getElementById('drop-area');
                const fileInput = document.getElementById('file-input');
                const resultDiv = document.getElementById('result');
                
                // Prevent default drag behaviors
                ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
                    dropArea.addEventListener(eventName, preventDefaults, false);
                });
                
                function preventDefaults(e) {
                    e.preventDefault();
                    e.stopPropagation();
                }
                
                // Highlight drop area when dragging over it
                ['dragenter', 'dragover'].forEach(eventName => {
                    dropArea.addEventListener(eventName, highlight, false);
                });
                
                ['dragleave', 'drop'].forEach(eventName => {
                    dropArea.addEventListener(eventName, unhighlight, false);
                });
                
                function highlight() {
                    dropArea.classList.add('highlight');
                }
                
                function unhighlight() {
                    dropArea.classList.remove('highlight');
                }
                
                // Handle dropped files
                dropArea.addEventListener('drop', handleDrop, false);
                
                function handleDrop(e) {
                    const dt = e.dataTransfer;
                    const files = dt.files;
                    
                    if (files.length > 0) {
                        handleFiles(files);
                    }
                }
                
                // Handle selected files
                fileInput.addEventListener('change', function() {
                    handleFiles(this.files);
                });
                
                function handleFiles(files) {
                    const file = files[0];
                    uploadFile(file);
                }
                
                function uploadFile(file) {
                    const formData = new FormData();
                    formData.append('file', file);
                    
                    fetch('/upload', {
                        method: 'POST',
                        body: formData
                    })
                    .then(response => response.json())
                    .then(data => {
                        if (data.error) {
                            alert(data.error);
                            return;
                        }
                        
                        // Display results
                        document.getElementById('prediction').textContent = data.prediction;
                        document.getElementById('confidence').textContent = (data.confidence * 100).toFixed(2) + '%';
                        document.getElementById('inference-time').textContent = data.inference_time.toFixed(3);
                        document.getElementById('result-image').src = 'data:image/jpeg;base64,' + data.image_base64;
                        
                        // Show result div
                        resultDiv.style.display = 'block';
                    })
                    .catch(error => {
                        console.error('Error:', error);
                        alert('An error occurred during upload.');
                    });
                }
            </script>
            {% endif %}
            
            <div style="margin-top: 40px; border-top: 1px solid #ccc; padding-top: 20px;">
                <p><a href="/about">About</a> | FreshHarvest Fruit Freshness Detection</p>
            </div>
        </div>
    </body>
    </html>
    """
    
    # About template
    about_html = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>About FreshHarvest</title>
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <style>
            body { font-family: Arial, sans-serif; margin: 0; padding: 20px; }
            .container { max-width: 800px; margin: 0 auto; }
            h1 { color: #2C8C3C; }
            a { color: #2C8C3C; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>About FreshHarvest</h1>
            
            <p>FreshHarvest is a fruit freshness detection system using computer vision and deep learning.</p>
            
            <h2>How it works</h2>
            <p>The system uses a convolutional neural network (CNN) trained on thousands of images of fresh and spoiled fruits.
               When you upload an image, the model analyzes it and predicts whether the fruit is fresh or spoiled.</p>
            
            <h2>Supported Fruits</h2>
            <ul>
                <li>Banana</li>
                <li>Lemon</li>
                <li>Lulo</li>
                <li>Mango</li>
                <li>Orange</li>
                <li>Strawberry</li>
                <li>Tamarillo</li>
                <li>Tomato</li>
            </ul>
            
            <p><a href="/">Back to home</a></p>
        </div>
    </body>
    </html>
    """
    
    # 404 template
    not_found_html = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Page Not Found - FreshHarvest</title>
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <style>
            body { font-family: Arial, sans-serif; margin: 0; padding: 20px; }
            .container { max-width: 800px; margin: 0 auto; text-align: center; padding-top: 50px; }
            h1 { color: #2C8C3C; }
            a { color: #2C8C3C; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>404 - Page Not Found</h1>
            <p>The page you're looking for doesn't exist.</p>
            <p><a href="/">Go back to home</a></p>
        </div>
    </body>
    </html>
    """
    
    # 500 template
    server_error_html = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Server Error - FreshHarvest</title>
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <style>
            body { font-family: Arial, sans-serif; margin: 0; padding: 20px; }
            .container { max-width: 800px; margin: 0 auto; text-align: center; padding-top: 50px; }
            h1 { color: #2C8C3C; }
            a { color: #2C8C3C; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>500 - Server Error</h1>
            <p>Something went wrong on our end. Please try again later.</p>
            <p><a href="/">Go back to home</a></p>
        </div>
    </body>
    </html>
    """
    
    # Write templates to files
    with open(templates_dir / 'index.html', 'w') as f:
        f.write(index_html)
    
    with open(templates_dir / 'about.html', 'w') as f:
        f.write(about_html)
    
    with open(templates_dir / '404.html', 'w') as f:
        f.write(not_found_html)
    
    with open(templates_dir / '500.html', 'w') as f:
        f.write(server_error_html)
    
    return "Templates generated!"

if __name__ == '__main__':
    # Generate templates if they don't exist
    if not os.path.exists('templates'):
        generate_templates()
    
    # Run the app
    app.run(host='0.0.0.0', port=5000, debug=True)