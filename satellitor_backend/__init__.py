from flask import Flask
from flask_cors import CORS
from ultralytics import YOLO
import os
import ee
import json
import tempfile


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, 'best.pt')
model = YOLO(model_path)
app = Flask(__name__)
CORS(app)
crops = json.load(open(os.path.join(BASE_DIR, 'crops.json'), 'r'))


google_card_json = os.getenv('GOOGLE_CREDENTIALS')

if not google_card_json:
    raise Exception("No Cred. is Set")

with tempfile.NamedTemporaryFile(mode='w+', delete=False) as temp_cred_file:
    temp_cred_file.write(google_card_json)
    temp_cred_file_path = temp_cred_file.name

# Initialize EE with the service account email and the temporary credentials file
credentials = ee.ServiceAccountCredentials(
    "earth-engine-access@premium-buckeye-310022.iam.gserviceaccount.com",
    temp_cred_file_path
)
ee.Initialize(credentials)
print("done 7")

from satellitor_backend import routes

