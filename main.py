# Firebase init
import firebase_admin
from firebase_admin import credentials
from firebase_admin import db
from firebase_admin import firestore
from google.cloud import storage

# Package tools for machine learning init
import numpy as np
from PIL import Image
import tensorflow as tf
import os
import h5py
from flask import Flask, request, jsonify

# Install misc
from dotenv import load_dotenv
import pytz
import urllib.request
import datetime
import os

load_dotenv()

firebase_config = {
    "apiKey": os.getenv("API_KEY"),
    "authDomain": os.getenv("AUTH_DOMAIN"),
    "projectId": os.getenv("PROJECT_ID"),
    "storageBucket": os.getenv("STORAGE_BUCKET"),
    "messagingSenderId": os.getenv("MESSAGING_SENDER_ID"),
    "appId": os.getenv("APP_ID")
}

# Initialize Firebase SDK
cred = credentials.Certificate('./db-setup/firebase-key.json')
firebase_admin.initialize_app(cred)

db = firestore.client()

app = Flask(__name__)
bucket_name = "xdetect-img-profile"
class_mapping = ["Mass", "Nodule", "Normal", "Pneumonia", "Tuberculosis"]

additional_info = {
        "Normal": {
        "description": "Kondisi normal tanpa adanya kelainan pada paru-paru.",
        "symptoms": [],
        "nextSteps": []
    },
    "Mass": {
        "description": "Adanya massa atau tumor di paru-paru, yang bisa bersifat jinak atau ganas.",
        "symptoms": ["Batuk", "Sesak napas", "Nyeri dada"],
        "nextSteps": [
        "Konsultasikan dengan spesialis untuk evaluasi lebih lanjut",
        "Tambahan tes mungkin diperlukan",
        "Ikuti rencana perawatan yang direkomendasikan"
        ]
    },
    "Nodule": {
        "description": "Adanya pertumbuhan atau benjolan kecil yang tidak normal di paru-paru, yang bisa bersifat jinak atau ganas.",
        "symptoms": ["Batuk", "Sesak napas", "Nyeri dada"],
        "nextSteps": [
        "Konsultasikan dengan spesialis untuk evaluasi lebih lanjut",
        "Tambahan tes mungkin diperlukan",
        "Ikuti rencana perawatan yang direkomendasikan"
        ]
    },
    "Pneumonia": {
        "description": "Infeksi pada satu atau kedua paru-paru, yang bisa menyebabkan peradangan dan gejala pernapasan.",
        "symptoms": ["Demam", "Batuk", "Sesak napas", "Nyeri dada"],
        "nextSteps": [
        "Cari perhatian medis untuk diagnosis dan pengobatan",
        "Konsumsi obat yang diresepkan",
        "Istirahat yang cukup dan tetap terhidrasi"
        ]
    },
    "Tuberculosis": {
        "description": "Infeksi bakteri yang terutama mempengaruhi paru-paru, tetapi juga dapat mempengaruhi bagian tubuh lainnya.",
        "symptoms": ["Batuk yang tidak kunjung sembuh", "Nyeri dada", "Kelelahan", "Demam", "Penurunan berat badan"],
        "nextSteps": [
        "Cari perhatian medis untuk diagnosis dan pengobatan",
        "Ikuti jadwal pengobatan yang diresepkan",
        "Ambil tindakan pencegahan untuk mencegah penyebaran infeksi kepada orang lain"
        ]
    },
}

recommendation = {
    "action": "Mohon segera lakukan konsultasi dengan dokter spesialis",
    "message": "null",
}

def getXrayClass(image_path, model_h5_path, class_mapping):

    image = Image.open(image_path)
    image = image.convert("RGB")  # Convert image to RGB mode
    image = image.resize((224, 224))  # Resize the image to match the expected input shape of the model
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)

    # Verify the model.h5 file exists
    if not os.path.isfile(model_h5_path):
        print("Model file does not exist:", model_h5_path)
        return None

    # Check the contents of the model.h5 file
    with h5py.File(model_h5_path, "r") as f:
        if "model_config" not in f.attrs.keys():
            print("Invalid model.h5 file:", model_h5_path)
            return None

    # Load the trained model
    model = tf.keras.models.load_model(model_h5_path, compile=False)
    model.compile(optimizer=tf.keras.optimizers.legacy.Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

    # Perform inference
    predictions = model.predict(image)
    print(predictions)
    predicted_class_index = np.argmax(predictions)
    predicted_class = class_mapping[predicted_class_index]

    print(predicted_class)

    return predictions, predicted_class

# Function for running x-ray prediction
def runXrayClassify(image_url, model_h5_path):
    # Download the image from the provided URL
    image_path = 'temp.jpg'
    urllib.request.urlretrieve(image_url, image_path)

    # Call the getXrayClass function
    predictions, predicted_class = getXrayClass(image_path, model_h5_path, class_mapping)

    # Remove the temporary image file
    os.remove(image_path)

    return predictions, predicted_class

# Function for upload files to bucket
def uploadXray(bucket_name, file_name, file):
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(file_name)

    blob.upload_from_file(file, content_type='application/octet-stream')

    blob.make_public()

    public_url = blob.public_url

    return public_url

@app.route('/predict/<uid>', methods=['POST'])
def post_xray_images(uid):
    try:
        if len(request.files) == 0:
            response = jsonify({
                'status': 'Failed',
                'message': 'Tidak ada file yang ditambahkan'
            })
            response.status_code = 400
            return response

        file = next(iter(request.files.values()))
        file_name = file.filename

        public_url = uploadXray(bucket_name, file_name, file)

        predictions, predicted_class = runXrayClassify(public_url, 'modelxdetect.h5')
        prediction_result = {}
        for index, prediction in enumerate(predictions[0]):
            prediction_result[class_mapping[index]] = f"{float(prediction) * 100}%"

        # Adjust timestamp to client's time zone
        client_timezone = pytz.timezone('Asia/Jakarta')  # Replace with the appropriate time zone
        client_timestamp = datetime.datetime.now(client_timezone)

        doc_ref = db.collection('users2').document(uid)
        doc_ref.update({
            'history': firestore.ArrayUnion([{
                'type': 'Chest X-Ray Detection',
                'datetime': client_timestamp,
                'predicted_class': predicted_class,
                'detection_img': public_url,
            }])
        })

        max_label = {
            'label': predicted_class,
            'percentage': prediction_result[predicted_class]
        }

        response = jsonify({
            'status': 'Success',
            'message': 'Deteksi penyakit berhasil',
            'detection_img': public_url,
            'class': predicted_class,
            'predictions': prediction_result,
            'maxLabel': max_label,
            'created': client_timestamp.strftime("%m/%d/%Y, %I:%M:%S %p"),
            'additionalInfo': additional_info.get(predicted_class, {}),
            'recommendation': recommendation,
        })
        response.status_code = 200
        return response

    except Exception as error:
        print(error)

        response = jsonify({
            'status': 'Failed',
            'message': 'An internal server error occurred',
            'error': str(error)
        })
        response.status_code = 500
        return response
    
@app.route('/predict/history/<uid>', methods=['GET'])
def getHistoryPrediction(uid):
    try:
        # Retrieve the user document
        doc_ref = db.collection('users2').document(uid)
        doc = doc_ref.get()

        if doc.exists:
            # Retrieve the history data from the document
            history = doc.to_dict().get('history', [])

            formatted_history = []
            for item in history:
                datetime_value = item['datetime']
                datetime_utc = datetime_value.replace(tzinfo=pytz.UTC)
                datetime_local = datetime_utc.astimezone(pytz.timezone('Asia/Jakarta'))
                formatted_datetime = datetime_local.strftime("%d %B, %Y at %H:%M:%S")
                item['datetime'] = formatted_datetime
                formatted_history.append(item)

            # Reverse the order of the history list
            formatted_history.reverse()

            response = jsonify({
                'status': 'Success',
                'message': 'History prediksi xray telah didapat',
                'data': formatted_history
            })
            response.status_code = 200
            return response
        else:
            response = jsonify({
                'status': 'Failed',
                'message': 'User tidak ada dalam database'
            })
            response.status_code = 404
            return response

    except Exception as error:
        print(error)  # Log the error for debugging purposes

        response = jsonify({
            'status': 'Failed',
            'message': 'An internal server error occurred',
            'error': str(error)
        })
        response.status_code = 500
        return response


# Running Flask
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8080)