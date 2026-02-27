# app.py
from flask import Flask, render_template, Response, request, jsonify, send_file
import cv2
import easyocr
import re
import numpy as np
import base64
from PIL import Image
import io
import os
from datetime import datetime
import threading
import time

app = Flask(__name__)

# Global variables
reader = None
camera = None
current_frame = None
detection_history = []
camera_active = False
selected_roi = None
captured_frame = None  # Store captured frame for processing

# Initialize EasyOCR
def init_ocr():
    global reader
    if reader is None:
        print("📦 Loading EasyOCR...")
        reader = easyocr.Reader(['en'])
        print("✅ EasyOCR ready!")

# Camera functions
def init_camera():
    global camera, camera_active
    if camera is None:
        camera = cv2.VideoCapture(0)
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        camera_active = True
        print("✅ Camera initialized")

def get_frame():
    global camera, current_frame
    if camera and camera.isOpened():
        ret, frame = camera.read()
        if ret:
            current_frame = frame.copy()
            # Add overlay text
            cv2.putText(frame, "Printing Press Counter Reader", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"Time: {datetime.now().strftime('%H:%M:%S')}", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            ret, jpeg = cv2.imencode('.jpg', frame)
            return jpeg.tobytes()
    return None

def read_numbers_from_image(image):
    """Read numbers from image"""
    results = []
    
    # Direct OCR
    result = reader.readtext(image)
    
    if result:
        for detection in result:
            text = detection[1]
            confidence = detection[2]
            numbers = re.sub(r'[^0-9]', '', text)
            
            if numbers:
                results.append({
                    'text': numbers,
                    'confidence': float(confidence),
                    'method': 'direct'
                })
    
    # If no results, try preprocessing
    if not results:
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Try different preprocessing methods
        # Method 1: Threshold
        _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
        scaled = cv2.resize(thresh, None, fx=2, fy=2)
        
        result2 = reader.readtext(scaled)
        
        for detection in result2:
            text = detection[1]
            numbers = re.sub(r'[^0-9]', '', text)
            if numbers:
                results.append({
                    'text': numbers,
                    'confidence': float(detection[2]),
                    'method': 'enhanced'
                })
        
        # Method 2: Adaptive threshold
        if not results:
            adaptive = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                           cv2.THRESH_BINARY, 11, 2)
            scaled_adaptive = cv2.resize(adaptive, None, fx=2, fy=2)
            
            result3 = reader.readtext(scaled_adaptive)
            
            for detection in result3:
                text = detection[1]
                numbers = re.sub(r'[^0-9]', '', text)
                if numbers:
                    results.append({
                        'text': numbers,
                        'confidence': float(detection[2]),
                        'method': 'adaptive'
                    })
    
    return results

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    def generate():
        while True:
            frame = get_frame()
            if frame:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            time.sleep(0.03)  # ~30 FPS
    return Response(generate(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/capture', methods=['POST'])
def capture_image():
    """Capture current frame for processing"""
    global captured_frame, current_frame
    
    if current_frame is None:
        return jsonify({'error': 'No frame available'}), 400
    
    # Store the captured frame
    captured_frame = current_frame.copy()
    
    # Save captured image
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    capture_filename = f'static/captures/capture_{timestamp}.jpg'
    os.makedirs('static/captures', exist_ok=True)
    cv2.imwrite(capture_filename, captured_frame)
    
    return jsonify({
        'status': 'success',
        'message': 'Image captured',
        'image_path': capture_filename,
        'timestamp': timestamp
    })

@app.route('/process_full_image', methods=['POST'])
def process_full_image():
    """Process the entire captured image"""
    global captured_frame, detection_history
    
    if captured_frame is None:
        return jsonify({'error': 'No captured image available. Please capture first.'}), 400
    
    data = request.json
    use_full_image = data.get('use_full_image', True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    if use_full_image:
        # Process entire image
        image_to_process = captured_frame
        roi_filename = f'static/captures/full_{timestamp}.jpg'
        cv2.imwrite(roi_filename, image_to_process)
    else:
        # Use selected ROI
        if selected_roi is None:
            return jsonify({'error': 'No ROI selected. Please select a region first.'}), 400
        
        x, y, w, h = selected_roi
        image_to_process = captured_frame[y:y+h, x:x+w]
        roi_filename = f'static/captures/roi_{timestamp}.jpg'
        cv2.imwrite(roi_filename, image_to_process)
    
    # Read numbers
    results = read_numbers_from_image(image_to_process)
    
    # Save to history
    detection_entry = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'image_path': roi_filename,
        'results': results,
        'method': 'full_image' if use_full_image else 'roi_only',
        'roi_coords': selected_roi if not use_full_image else None
    }
    detection_history.append(detection_entry)
    
    # Keep only last 50 detections
    if len(detection_history) > 50:
        detection_history.pop(0)
    
    return jsonify({
        'status': 'success',
        'results': results,
        'image_path': roi_filename,
        'timestamp': detection_entry['timestamp']
    })

@app.route('/save_roi', methods=['POST'])
def save_roi():
    """Save ROI coordinates"""
    global selected_roi
    data = request.json
    selected_roi = (data['x'], data['y'], data['width'], data['height'])
    return jsonify({'status': 'success', 'roi': selected_roi})

@app.route('/get_current_frame', methods=['GET'])
def get_current_frame():
    """Get current frame as base64 for ROI selection"""
    global current_frame
    
    if current_frame is None:
        return jsonify({'error': 'No frame available'}), 400
    
    # Encode frame to base64
    _, buffer = cv2.imencode('.jpg', current_frame)
    img_base64 = base64.b64encode(buffer).decode('utf-8')
    
    return jsonify({
        'status': 'success',
        'image': img_base64
    })

@app.route('/get_history')
def get_history():
    return jsonify({'history': detection_history})

@app.route('/clear_history', methods=['POST'])
def clear_history():
    global detection_history
    detection_history = []
    return jsonify({'status': 'success'})

@app.route('/reset_roi', methods=['POST'])
def reset_roi():
    global selected_roi
    selected_roi = None
    return jsonify({'status': 'success'})

if __name__ == '__main__':
    init_ocr()
    init_camera()
    app.run(debug=True, host='0.0.0.0', port=5000)