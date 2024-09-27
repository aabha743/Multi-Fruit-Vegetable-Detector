from flask import Flask, request, render_template
import torch
import cv2
import os

model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

app = Flask(__name__)


if not os.path.exists('uploads'):
    os.makedirs('uploads')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return {"error": "No file provided"}, 400

    file = request.files['file']

    
    image_path = os.path.join('uploads', file.filename)
    file.save(image_path)

    
    img = cv2.imread(image_path)
    if img is None:
        return {"error": "Unable to load image. Please upload a valid image file."}, 400

    
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    
    results = model(img_rgb)

    
    predictions = results.pred[0].cpu().numpy()

    output = {}
    for pred in predictions:
        x1, y1, x2, y2, confidence, class_id = pred
        label = results.names[int(class_id)]  
        if label in output:
            output[label] += 1  
        else:
            output[label] = 1  

  
    return render_template('index.html', results=output)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
