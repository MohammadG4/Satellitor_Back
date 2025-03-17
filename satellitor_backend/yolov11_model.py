from collections import defaultdict
from types import NoneType

import requests
from satellitor_backend import model,crops
import cv2
import numpy as np
import ee
from ultralytics import YOLO,SAM
import os




class_colors = {
    0: (0, 0, 0),  # background
    1: (0, 255, 0), # agriculture
    2: (255, 255, 255),# Barren
    3: (0, 127, 0), # Forest
    4: (0, 0, 255),  # Urban
    5: (255, 0, 0)    # Water
}


# ============================================================

def yolo_mask(img, model=model):
        results = model(img)
        masks = results[0].masks
        return results[0]

# ============================================================

def assign_colors(mask_img):
    """
    Assigns specific colors to pixels in the mask image based on their RGB values.

    Parameters:
        mask_img (np.array): The input mask image.

    Returns:
        np.array: Updated mask image with assigned colors.
    """
    # Reshape image to 2D (flatten pixels)
    height, width, _ = mask_img.shape

    # Loop through every pixel in the image
    # for y in range(height):
    #     for x in range(width):
    #         pixel = mask_img[y, x]  # Get pixel RGB values
    #
    #         if pixel[0] < 10 and pixel[1] < 10 and pixel[2] < 10:
    #             mask_img[y, x] = class_colors[0]
    #         elif pixel[0] < 10 and pixel[1] > 240 and pixel[2] < 10:
    #             mask_img[y, x] = class_colors[1]
    #         elif pixel[0] > 240 and pixel[1] > 240 and pixel[2] > 240:
    #             mask_img[y, x] = class_colors[2]
    #         elif pixel[0] < 10 and pixel[1] == 127 and pixel[2] < 10:
    #             mask_img[y, x] = class_colors[3]
    #         elif pixel[0] < 10 and pixel[1] < 10 and pixel[2] > 240:
    #             mask_img[y, x] = class_colors[4]
    #         elif pixel[0] > 240 and pixel[1] < 10 and pixel[2] < 10:
    #             mask_img[y, x] = class_colors[5]
    for y in range(height):
        for x in range(width):
            pixel = mask_img[y, x]
            print(f"Pixel at ({x},{y}): {pixel}")  # Debugging line

    return mask_img

# ============================================================

def get_mask(img_path,output_path, model=model):
    img = cv2.imread(img_path)
    if img is None:
        print("Error: Image not found!")
        return

    mask_img = np.zeros_like(img)

    results = model.predict(img)

    for result in results:
        if result.masks is not None:
            masks = result.masks.data.cpu().numpy()
            class_ids = result.boxes.cls.cpu().numpy().astype(int)

            for i, mask in enumerate(masks):
                mask = cv2.resize(mask, (img.shape[1], img.shape[0]))
                mask = (mask > 0.5).astype(np.uint8)
                color = class_colors.get(class_ids[i], (0, 0, 0))
                mask_img[mask == 1] = color

    cv2.imwrite(output_path, mask_img)
    print("Mask processing completed!")
    return mask_img

# ============================================================

def get_Percentage(mask_img,show_in_console):

    px = mask_img.reshape(-1, 3)
    class_count = defaultdict(int)

    for pixel in px:
        if pixel[0] < 10 and pixel[1] < 10 and pixel[2] < 10:
            class_count["Background"] += 1
        elif pixel[0] < 10 and pixel[1] > 240 and pixel[2] < 10:
            class_count["Agriculture"] += 1
        elif pixel[0] > 240 and pixel[1] > 240 and pixel[2] > 240:
            class_count["Barren"] += 1
        elif pixel[0] < 10 and pixel[1] == 127 and pixel[2] < 10:
            class_count["Forest"] += 1
        elif pixel[0] < 10 and pixel[1] < 10 and pixel[2] > 240:
            class_count["Urban"] += 1
        elif pixel[0] > 240 and pixel[1] < 10 and pixel[2] < 10:
            class_count["Water"] += 1

    class_percentage = {"Background":0.0,"Agriculture":0.0,"Barren":0.0,"Forest":0.0,"Urban":0.0,"Water":0.0}


    for class_name, count in class_count.items():
        class_percentage[class_name] = round(count / px.shape[0],3)
        if show_in_console:
            print(f"{class_name}: {round(count / px.shape[0] * 100,1)}%")

    return class_percentage

# ============================================================

def rgb_to_class_index(img_mask, color_map, tolerance=10):
    # Get image dimensions
    h, w, _ = img_mask.shape
    class_mask = np.zeros((h, w), dtype=np.uint8)

    img_mask = img_mask.astype(np.int16)

    for class_idx, (r_ref, g_ref, b_ref) in color_map.items():
        # Define lower and upper bounds
        lower_bound = np.array([max(0, r_ref - tolerance), max(0, g_ref - tolerance), max(0, b_ref - tolerance)])
        upper_bound = np.array([min(255, r_ref + tolerance), min(255, g_ref + tolerance), min(255, b_ref + tolerance)])

        class_range = (
                (img_mask[:, :, 0] >= lower_bound[0]) & (img_mask[:, :, 0] <= upper_bound[0]) &
                (img_mask[:, :, 1] >= lower_bound[1]) & (img_mask[:, :, 1] <= upper_bound[1]) &
                (img_mask[:, :, 2] >= lower_bound[2]) & (img_mask[:, :, 2] <= upper_bound[2])
        )

        class_mask[class_range] = class_idx

    return class_mask


def get_fragmentation(img_mask, class_colors=class_colors):
    class_mask = rgb_to_class_index(img_mask, class_colors)

    fragmentation_index = {}
    normalized_FI = {}

    for class_idx in class_colors.keys():
        # Ignore background
        if class_idx == 0:
            continue

        # Create a binary mask for the current class
        binary_mask = (class_mask == class_idx).astype(np.uint8)

        # -------------------------------------------------------------------------------------

        area = np.sum(binary_mask)

        if area == 0:
            fragmentation_index[class_idx] = 0
            normalized_FI[class_idx] = 0
            continue

        # -------------------------------------------------------------------------------------

        # Find connected components
        num_labels, _ = cv2.connectedComponents(binary_mask)

        # -------------------------------------------------------------------------------------
        normalized_FI[class_idx] = float((num_labels - 1) / area)
        # -------------------------------------------------------------------------------------

        # save FI (subtract 1 because background is counted)
        fragmentation_index[class_idx] = num_labels - 1

    return fragmentation_index,normalized_FI


# ============================================================

def detect_edges(mask_img ,output_path):
    if len(mask_img.shape) > 2:
        gray = cv2.cvtColor(mask_img, cv2.COLOR_BGR2GRAY)
    else:
        gray = mask_img

    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)

    cv2.imwrite(output_path, edges)
    return edges


# ============================================================

def get_land_properties(lat,long):

    print("Getting land points...")
    point = ee.Geometry.Point([long,lat])
    print("Land Points have been retrieved")


    print("Getting land PH...")
    ph_dataset = ee.Image("OpenLandMap/SOL/SOL_PH-H2O_USDA-4C1A2A_M/v02")

    bands=["b0","b10","b30","b60","b100"]
    ph_value=0
    b_num=0
    for band in bands:
        try:
            ph_value += ph_dataset.sample(point, scale=30).first().get(band).getInfo()
            b_num += 1
        except:
            continue
    if b_num == 0:
        ph_value = -1
    else:
        ph_value = ph_value/(b_num*10)
    print("Land PH calculated!")


    print("Getting land Precipitation...")
    d= ee.Image("OpenLandMap/CLM/CLM_PRECIPITATION_SM2RAIN_M/v01")

    monthly_bands = ['jan', 'feb', 'mar', 'apr', 'may', 'jun','jul', 'aug', 'sep', 'oct', 'nov', 'dec']

    annual_precip = d.select(monthly_bands).reduce(ee.Reducer.sum())

    sample = annual_precip.sample(region=point, scale=1000).first()

    annual_mm = sample.get('sum').getInfo()
    print(f"Annual precipitation (mm/year): {annual_mm}")

    nasa_api = f"https://power.larc.nasa.gov/api/temporal/climatology/point?parameters=T2M,RH2M&latitude={lat}&longitude={long}&format=JSON&community=ag"
    climate_data = requests.get(nasa_api).json()


    temperature = climate_data["properties"]["parameter"]["T2M"]["ANN"]
    humidity = climate_data["properties"]["parameter"]["RH2M"]["ANN"]

    print(f"Location: ({lat}, {long})")
    print(f"Soil pH (0-5cm depth): {ph_value}")
    print(f"Annual Avg Temperature: {temperature}°C")
    print(f"Annual Avg Humidity: {humidity}%")
    print(f"Annual Rainfall: {annual_mm} mm")

    return  ph_value, temperature, humidity, annual_mm

# ============================================================

def sam_refining(img_path):

    return None
# ============================================================

def get_crops(ph,temp,rainfall):
    best_crops=[]
    for crop_name,crop_prop in crops.items():
        isPh = False
        isTemp = False
        isRainfall = False
        crop_notes = ""

        if not (crop_prop['temp_min'] <= temp <= crop_prop['temp_max']):
            continue
        isTemp = True


        if ph != -1:
            if crop_prop['ph_min'] <= ph <= crop_prop['ph_max']:
                isPh = True
            else:
                continue
        else:
            crop_notes += " pH not provided in this land. "

        if crop_prop['rainfall_min'] <= rainfall <= crop_prop['rainfall_max']:
            isRainfall = True
        else:
            crop_notes += f" Needs attention to irrigation. Optimal rainfall: {crop_prop['rainfall_opt_min']} mm/year to {crop_prop['rainfall_opt_max']} mm/year. "

        best_crops.append({
                "crop_name":crop_name,
                "isTemp":isTemp,
                "isPh":isPh,
                "isRainfall":isRainfall,
                "crop_notes":crop_notes.strip(),
                "crop_data":crop_prop
            })
    return best_crops

# ============================================================

def get_best_crops(ph,temp,rainfall):
    best_crops=[]
    for crop_name,crop_prop in crops.items():
        isPh = False
        isTemp = False
        isRainfall = False
        crop_notes = ""

        if not (crop_prop['temp_opt_min'] <= temp <= crop_prop['temp_opt_max']):
            continue
        isTemp = True


        if ph != -1:
            if crop_prop['ph_opt_min'] <= ph <= crop_prop['ph_opt_max']:
                isPh = True
            else:
                continue
        else:
            crop_notes += " pH not provided in this land. "

        if crop_prop['rainfall_opt_min'] <= rainfall <= crop_prop['rainfall_opt_max']:
            isRainfall = True
        else:
            crop_notes += f" Needs attention to irrigation. Optimal rainfall: {crop_prop['rainfall_opt_min']} mm/year to {crop_prop['rainfall_opt_max']} mm/year. "

        best_crops.append({
                "crop_name":crop_name,
                "isTemp":isTemp,
                "isPh":isPh,
                "isRainfall":isRainfall,
                "crop_notes":crop_notes.strip(),
                "crop_data":crop_prop
            })
    return best_crops




# Overlay this mask onto the result
#                 # Only replace pixels where this mask exists, preserving previous class assignments
#                 non_zero_indices = np.where(binary_mask == 1)
#                 mask_img[non_zero_indices] = colored_mask[non_zero_indices]