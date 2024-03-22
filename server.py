from flask import Flask, request, jsonify
from app.model import build_model
from app.preprocess import preprocess_data
import numpy as np
import os

app = Flask(__name__)

# Load the pre-trained model
model = build_model()
model.load_weights('pretrained_model.h5')


@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    if file:
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)

        X_new, _, label_encoder = preprocess_data(filepath)
        # Print the shape of input data
        print("Shape of input data:", X_new.shape)
        predictions = model.predict(X_new)

        # Decode the predicted classes
        predicted_classes = label_encoder.inverse_transform(np.argmax(predictions, axis=1))

        results = [{'Filename': file.filename, 'Predicted_AQI_Class': predicted_class} for predicted_class in
                   predicted_classes]

        return jsonify({'predictions': results})
    else:
        return jsonify({'error': 'Failed to process file'})


if __name__ == '__main__':
    app.config['UPLOAD_FOLDER'] = 'uploads/'
    app.run(debug=True)
