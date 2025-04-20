from flask import Flask, request, render_template, jsonify, redirect, url_for, send_from_directory
from tensorflow.keras.models import load_model
import numpy as np
import cv2
import os
import tensorflow as tf
import json
import time
import logging

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = Flask(__name__, static_folder='static')

# Load the best available model
model_path = 'pneumonia_model_custom.h5' if os.path.exists('pneumonia_model_custom.h5') else None
if model_path:
    try:
        model = load_model(model_path)
        logger.debug(f"Model loaded successfully from {model_path}")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        model = None
else:
    logger.error("Model not found! Ensure pneumonia_model_custom.h5 is in the directory.")
    model = None

# Serve static files (e.g., heatmap.png)
@app.route('/static/<path:filename>')
def serve_static(filename):
    return send_from_directory('static', filename)

def generate_heatmap(model, img_array, last_conv_layer_name='conv2d_6'):
    grad_model = tf.keras.models.Model([model.inputs], [model.get_layer(last_conv_layer_name).output, model.output])
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        loss = predictions[:, 0]
    grads = tape.gradient(loss, conv_outputs)[0]
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    heatmap = tf.reduce_mean(tf.multiply(pooled_grads, conv_outputs), axis=-1)
    # Ensure heatmap is a 2D array and convert to NumPy
    heatmap = heatmap.numpy()  # Convert Tensor to NumPy array
    if len(heatmap.shape) > 2:
        heatmap = np.squeeze(heatmap, axis=0)  # Remove batch dimension if present
    if len(heatmap.shape) == 1:
        heatmap = np.reshape(heatmap, (heatmap.shape[0], 1))  # Ensure 2D for resize
    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap + 1e-10)  # Avoid division by zero
    # Resize and cast to uint8
    heatmap = cv2.resize(heatmap, (224, 224))
    heatmap = np.uint8(255 * heatmap)  # Cast to 8-bit unsigned integer
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    # Cast the base image to uint8 before addWeighted
    base_img = np.uint8(255 * img_array[0, ..., 0])  # Scale float32 [0, 1] to [0, 255] and cast
    base_img_bgr = cv2.cvtColor(base_img, cv2.COLOR_GRAY2BGR)
    superimposed_img = cv2.addWeighted(base_img_bgr, 0.6, heatmap, 0.4, 0)
    heatmap_path = os.path.join('static', f'heatmap_{int(time.time())}.png')
    cv2.imwrite(heatmap_path, superimposed_img)
    logger.debug(f"Heatmap saved to {heatmap_path}")
    return f'/static/{os.path.basename(heatmap_path)}'

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload')
def upload():
    return render_template('upload.html')

@app.route('/api/predict', methods=['POST'])
def predict():
    if not model:
        logger.error("Model is None, cannot predict.")
        return jsonify({'error': 'Model not loaded'}), 400
    if 'xray' not in request.files:
        logger.error("No file part in request.")
        return jsonify({'error': 'No file part'}), 400
    file = request.files['xray']
    age = int(request.form.get('age', 0))
    gender = request.form.get('gender', 'male')
    if file.filename == '':
        logger.error("No selected file.")
        return jsonify({'error': 'No selected file'}), 400
    if file:
        try:
            img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_GRAYSCALE)
            if img is None:
                logger.error("Failed to decode image.")
                return jsonify({'error': 'Invalid image format'}), 400
            img = cv2.resize(img, (224, 224))
            img_array = np.expand_dims(img, axis=(0, -1)) / 255.0
            prediction = model.predict(img_array)
            confidence = prediction[0][0] * 100 * (1 + min(age / 120, 0.2))
            if gender == 'female':
                confidence *= 0.98
            result = 'PNEUMONIA' if confidence / 100 > 0.5 else 'NORMAL'
            displayed_confidence = (1 - prediction[0][0]) * 100 if result == 'NORMAL' else confidence
            heatmap_path = generate_heatmap(model, img_array)
            logger.debug(f"Prediction: {result}, Confidence: {displayed_confidence:.2f}")
            return jsonify({'prediction': result, 'confidence': round(min(displayed_confidence, 100), 2), 'heatmap': heatmap_path})
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            return jsonify({'error': 'An error occurred. Please try again.'}), 500

@app.route('/dashboard')
def dashboard():
    try:
        with open('training_history.json', 'r') as f:
            history = json.load(f)
        epochs = list(range(1, len(history.get('accuracy', [0]*10)) + 1))
        accuracy = history.get('accuracy', [0]*10)
        val_accuracy = history.get('val_accuracy', [0]*10)
        logger.debug(f"Loaded accuracy: {accuracy}, val_accuracy: {val_accuracy}")
    except FileNotFoundError:
        epochs = list(range(1, 11))
        accuracy = [0] * 10
        val_accuracy = [0] * 10
        logger.warning("training_history.json not found, using default data.")
    prediction = request.args.get('prediction', 'N/A')
    confidence = request.args.get('confidence', 'N/A')
    heatmap = request.args.get('heatmap', 'https://via.placeholder.com/224x224.png?text=Heatmap')
    return render_template('dashboard.html', accuracy=accuracy, val_accuracy=val_accuracy, epochs=epochs,
                          prediction=prediction, confidence=confidence, heatmap=heatmap)

if __name__ == '__main__':
    if not os.path.exists('static'):
        os.makedirs('static')
    app.run(debug=True, host='0.0.0.0', port=5000)