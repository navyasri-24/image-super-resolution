from flask import Flask, render_template, request, redirect, url_for, flash
from werkzeug.utils import secure_filename

import os
import cv2
import base64
import numpy as np
from model import model, predict

app = Flask(__name__)

# Set the upload folder and allowed extensions
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load the SRCNN model
srcnn_model = model()
srcnn_model.load_weights('3051crop_weight_200.h5')

# Function to check if the file has allowed extension
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Route to render the home page
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part')
            return redirect(url_for('index'))
        file = request.files['file']
        if file.filename == '':
            flash('No selected file')
            return redirect(url_for('index'))
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            # Create 'uploads' directory if it doesn't exist
            if not os.path.exists(app.config['UPLOAD_FOLDER']):
                os.makedirs(app.config['UPLOAD_FOLDER'])
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            ref, degraded, output, scores = predict(filepath)
            # Encode the super-resolved image as base64
            _, buffer = cv2.imencode('.jpg', output)
            img_str = base64.b64encode(buffer).decode('utf-8')
            return render_template('result.html', filename=filename, scores=scores, super_resolved=img_str)
    return redirect(url_for('index'))


if __name__ == '__main__':
    app.run(debug=True)
