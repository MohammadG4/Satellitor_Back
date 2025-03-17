from satellitor_backend import app
from flask import request, jsonify, send_from_directory
import os
import uuid
from satellitor_backend.yolov11_model import get_mask, detect_edges, get_land_properties, get_best_crops, \
    get_Percentage, get_crops, get_fragmentation

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUTS_FOLDER = os.path.join(BASE_DIR, 'inputs')
OUTPUTS_FOLDER = os.path.join(BASE_DIR, 'outputs')
os.makedirs(INPUTS_FOLDER, exist_ok=True)
os.makedirs(OUTPUTS_FOLDER, exist_ok=True)



@app.route('/process', methods=['POST','GET'])
def process():
    try:
        data = request.form
        longitude = float(data.get('longitude'))
        latitude = float(data.get('latitude'))
        print(type(longitude), type(latitude))

        if longitude is None or latitude is None:
            return jsonify({"error": "No Longitude or Latitude uploaded"},400)
        if "image" not in request.files:
            return jsonify({"error": "No image uploaded"},400)

        image = request.files['image']

        unique_id = str(uuid.uuid4())
        input_path = os.path.join(INPUTS_FOLDER, f"{unique_id}_input.png")
        image.save(input_path)

        #getting mask
        mask_path = os.path.join(OUTPUTS_FOLDER, f"{unique_id}_mask.png")
        mask_img = get_mask(input_path,mask_path)

        #getting edges (boundaries)
        boundaries_path = os.path.join(OUTPUTS_FOLDER, f"{unique_id}_boundaries.png")
        boundaries_img = detect_edges(mask_img,boundaries_path)
        print("Done3")

        percentage=get_Percentage(mask_img,False)
        print("Done2")

        ph_value, temperature, humidity, annual_mm = get_land_properties(lat=latitude,long=longitude)
        print("Done1")

        best_crops=get_best_crops(ph=ph_value,temp=temperature,rainfall=annual_mm)
        normal_crops=get_crops(ph=ph_value,temp=temperature,rainfall=annual_mm)

        _ , NFI =get_fragmentation(img_mask=mask_img)

        return jsonify({
            "percentage":percentage,
            "ph": ph_value,
            "temperature": temperature,
            "humidity": humidity,
            "rainfall": annual_mm,
            "normal_crops": normal_crops,
            "best_crops": best_crops,
            "normalized_FI": NFI,

            'mask_image': f"/download/{unique_id}_mask.png",
            'boundaries_image': f"/download/{unique_id}_boundaries.png"
        })

    except Exception as e:
        return jsonify({"error": "Something went wrong","details" : str(e)},400)


@app.route("/download/<filename>", methods=['GET'])
def download(filename):
    try:
        path = os.path.join(OUTPUTS_FOLDER, filename)
        if not os.path.exists(path):
            return jsonify({"error": "File not found"},404)

        return send_from_directory(OUTPUTS_FOLDER, filename)



    except Exception as e:
        return jsonify({"error": "Error fetching the file","details" : str(e)},400)