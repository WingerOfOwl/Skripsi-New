import os
import cv2
import numpy as np
from flask import Flask, request, jsonify
from ultralytics import YOLO 
import pytesseract
from flask_cors import CORS

app = Flask(__name__)

CORS(app, resources={r"/*": {"origins": "*"}})

#sertifikat
model_model= os.path.join(os.path.dirname(os.path.abspath(__file__)), 'yolov8_trained_80_10_10 (coba).pt')

#surattugas
model_model_st = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'yolov8_surat_tugas_80_10_10.pt')


yolo_model = YOLO(model_model) 

yolo_model_st = YOLO(model_model_st) 


# Define label mapping
label_mapping = {
    0: 'juara',
    1: 'nama',
    2: 'nama_lomba'
}
# Define label mapping
label_mapping_st = {
    0: 'nomor_surat'
}


def detect_and_crop(image_path):
    image = cv2.imread(image_path)
    results = yolo_model(image)  
    
    cropped_images_with_labels = []
    
    for result in results:
        boxes = result.boxes.cpu().numpy()
        for box in boxes:
            confidence = box.conf[0]  
            
            if confidence > 0.35:
                r = box.xyxy[0].astype(int)  
                cropped_image = image[r[1]:r[3], r[0]:r[2]]
                label = int(box.cls[0])  # Get the label index
                cropped_images_with_labels.append((cropped_image, label))

    return cropped_images_with_labels

def detect_and_crop_st(image_path):
    image = cv2.imread(image_path)
    results = yolo_model_st(image)  
    
    cropped_images_with_labels_st = []
    
    for result in results:
        boxes = result.boxes.cpu().numpy()
        for box in boxes:
            confidence = box.conf[0]  
            
            if confidence > 0.35:
                r = box.xyxy[0].astype(int)  
                cropped_image = image[r[1]:r[3], r[0]:r[2]]
                label = int(box.cls[0])  # Get the label index
                cropped_images_with_labels_st.append((cropped_image, label))

    return cropped_images_with_labels_st

def extract_text_from_cropped_images(cropped_images_with_labels):
    extracted_texts_with_labels = []
    
    for cropped_image, label in cropped_images_with_labels:
        # Convert the image to RGB format as pytesseract expects it.
        rgb_image = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB)

        # Use pytesseract to extract text from the image.
        text = pytesseract.image_to_string(rgb_image)

        extracted_texts_with_labels.append((text.strip(), label))  # Append stripped text and label
    
    return extracted_texts_with_labels

def extract_text_from_cropped_images_st(cropped_images_with_labels):
    extracted_texts_with_labels_st = []
    
    for cropped_image, label in cropped_images_with_labels:
        # Convert the image to RGB format as pytesseract expects it.
        rgb_image_st = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB)

        # Use pytesseract to extract text from the image.
        text = pytesseract.image_to_string(rgb_image_st)

        extracted_texts_with_labels_st.append((text.strip(), label))  # Append stripped text and label
    
    return extracted_texts_with_labels_st

@app.route('/process', methods=['POST'])
def process_image():
    if 'image' not in request.files:
        return jsonify({"error": "No image provided."}), 400
    
    image_file = request.files['image']
    
    if not image_file:
        return jsonify({"error": "Invalid image."}), 400
    
    try:
        image_path = 'temp.jpg'
        
        with open(image_path, 'wb') as f:
            f.write(image_file.read())
        
        # Detect and crop area using YoloV8 
        cropped_images_with_labels = detect_and_crop(image_path)

        # If no images detected 
        if not cropped_images_with_labels:
            return jsonify({"error": "Tidak ada teks terdeteksi."}), 400

        # Extract texts using pytesseract 
        ocr_results_with_labels = extract_text_from_cropped_images(cropped_images_with_labels)

        # Create result dictionary based on labels 
        result_dict = {"juara": "", "nama": "", "nama_lomba": ""}

        # Directly assign texts to corresponding labels
        for text, label in ocr_results_with_labels:
            if label in label_mapping:
                result_dict[label_mapping[label]] = text

        return jsonify(result_dict)

    except Exception as e:
        print(f"Error processing the image: {e}")
        return jsonify({"error": str(e)}), 500
    
@app.route('/detect_st', methods=['POST'])
def process_image_st():
    if 'image' not in request.files:
        return jsonify({"error": "No image provided."}), 400
    
    image_file_st = request.files['image']
    
    if not image_file_st:
        return jsonify({"error": "Invalid image."}), 400
    
    try:
        image_path = 'temp.jpg'
        
        with open(image_path, 'wb') as f:
            f.write(image_file_st.read())
        
        # Detect and crop area using YoloV8 
        cropped_images_with_labels_st = detect_and_crop_st(image_path)

        # If no images detected 
        if not cropped_images_with_labels_st:
            return jsonify({"error": "Tidak ada teks terdeteksi."}), 400

        # Extract texts using pytesseract 
        ocr_results_with_labels_st = extract_text_from_cropped_images_st(cropped_images_with_labels_st)

        # Create result dictionary based on labels 
        result_dict_st = {"nomor_surat": ""}

        # Directly assign texts to corresponding labels
        for text, label in ocr_results_with_labels_st:
            if label in label_mapping_st:
                result_dict_st[label_mapping_st[label]] = text

        return jsonify(result_dict_st)

    except Exception as e:
        print(f"Error processing the image: {e}")
        return jsonify({"error": str(e)}), 500
    
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)