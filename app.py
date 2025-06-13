from flask import Flask, request, jsonify, render_template
from flask_cors import CORS  # Enable cross-origin requests
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import os

app = Flask(__name__)
CORS(app)  # Allow frontend to access backend API

# Ensure "uploads" folder exists
os.makedirs("uploads", exist_ok=True)

# Load the trained model with error handling
MODEL_PATH = os.path.abspath("replace with model path")  # Use absolute path
print(f"Loading model from: {MODEL_PATH}")

try:
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None  # Prevent crashes if model fails to load

# Function to preprocess image
def preprocess_image(image):
    try:
        print(f"Original image size: {image.size}")  # Debugging log
        image = image.resize((224, 224))  # Resize to match model input
        image = np.array(image) / 255.0  # Normalize pixel values
        print(f"Processed image shape: {image.shape}")  # Debugging log
        image = np.expand_dims(image, axis=0)  # (1, 224, 224, 3)
        return image
    except Exception as e:
        print(f"Error in image preprocessing: {e}")
        return None  # Return None if preprocessing fails

@app.route('/')
def home():
    return render_template("index.html")  # Serve the frontend

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Model failed to load. Check logs for details.'})

    try:
        # Check if file is uploaded
        if 'image' not in request.files:
            return jsonify({'error': 'No file uploaded'})

        file = request.files['image']
        image = Image.open(io.BytesIO(file.read()))  # Read image
        image = preprocess_image(image)  # Preprocess

        if image is None:
            return jsonify({'error': 'Image preprocessing failed'})

        # Predict manuscript year
        prediction = model.predict(image)  # Output shape: (1, num_classes)
        print(f"Raw prediction output: {prediction}")

        predicted_year_index = np.argmax(prediction[0])  # Get index for single image
        ##year_classes =  [1700,1492,1937,1290,1364]
        year_classes = [1364,1492,1937,1290,1700]
        predicted_year = year_classes[predicted_year_index]

        print(f"Predicted Year: {predicted_year}")
        return jsonify({'predicted_year': int(predicted_year)})
    except Exception as e:
        print(f"Prediction error: {e}")
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(port=5000, debug=False)
