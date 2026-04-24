import os, re, json
from datetime import datetime, timezone

from PIL import Image
from termcolor import colored
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
import cv2

from google import genai
from google.genai import types
#######################################################
#######################################################
def get_timestamp_from_filename(filepath):
    # Expected pattern: <epoch>_<YYYYMMDD>_<HHMMSS>_...
    filename = os.path.basename(filepath)
    match = re.match(r"^(\d+)_(\d{8})_(\d{6})_", filename)
    if match:
        epoch = int(match.group(1))
        from_epoch_utc = datetime.fromtimestamp(epoch, tz=timezone.utc)
        return str(epoch)
    else:
        raise ValueError(f"Filename '{filename}' does not contain a valid timestamp.")
#######################################################
#######################################################
def load_images(directory_path):
    images = []
    if not os.path.isdir(directory_path):
        print(f"Error: Directory not found at '{directory_path}'")
        return []

    print(f"Searching for images in: {directory_path}")
    for filename in os.listdir(directory_path):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.webp')):
            filepath = os.path.join(directory_path, filename)
            timestamp = get_timestamp_from_filename(filepath)
            timestamp_str = datetime.fromtimestamp(int(timestamp), tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S %Z")
            try:
                image = Image.open(filepath).convert("RGB") # Ensure consistent mode
                images.append((timestamp, image))
                print(colored(f"Loaded image: ", "blue"), timestamp_str, filename, end='\r')
            except Exception as e:
                print(f"Could not load image {filename}: {e}")
    if not images:
        print(f"No images found in the directory '{directory_path}'. Please check the path and file types.")
    print("")
    print(colored("Total images loaded:", "blue"), len(images))
    return images
#######################################################
#######################################################
def addBoundingBoxToImage(image, bbox, color=(255, 0, 0), width=2):
    from PIL import ImageDraw
    draw = ImageDraw.Draw(image)
    draw.rectangle(bbox, outline=color, width=width)
    return image
#######################################################
#######################################################
def cropImageToBoundingBox(image, bbox):
    return image.crop(bbox)
#######################################################
#######################################################
def extractWaterGauge(image, bbox):

    #sideband region definition
    delta = np.array((20, 0, 20, 0))*4

    #smooching kernel for the signal region
    kernel = np.ones(10)/10

    #hardcoded pole start position in pixels
    poleEnd = 140

    #hardcoded middle of the water gauge in pixels 
    middlePos = 250

    #size of the window for which the intensity increase/decrease is evaluated
    windowSize = 10

    signalRegion = np.array(cropImageToBoundingBox(image, bbox))
    sidebandRegionA = np.array(cropImageToBoundingBox(image, bbox+delta))
    sidebandRegionB = np.array(cropImageToBoundingBox(image, bbox-delta))
    sidebandRegion = (sidebandRegionA + sidebandRegionB)//2

    signalRegion = signalRegion[:,:,:]
    sidebandRegion = sidebandRegion[:,:,:]

    signalRegion = np.mean(signalRegion, axis=(1,2))
    sidebandRegion = np.mean(sidebandRegion, axis=(1,2))
    signalRegion /= sidebandRegion
    signalRegion -= np.min(signalRegion)
    signalRegion = np.convolve(signalRegion, kernel, mode='same')
    signalRegion /= np.max(signalRegion)

    gradient = np.gradient(signalRegion)

    delta = 0
    tmpPos = middlePos
    startPos = 0
    endPos = len(signalRegion)-1
    while tmpPos-windowSize>0:
        deltaTmp = (signalRegion[tmpPos] - signalRegion[tmpPos-windowSize])
        #print(tmpPos, deltaTmp)
        if deltaTmp > delta:
            delta = deltaTmp
            startPos = tmpPos
        tmpPos -= windowSize

    tmpPos = middlePos
    delta = 0
    while tmpPos+windowSize<len(signalRegion):
        deltaTmp = -(signalRegion[tmpPos+windowSize] - signalRegion[tmpPos])
        #print(tmpPos, deltaTmp)
        if deltaTmp > delta:
            delta = deltaTmp
            endPos = tmpPos
        tmpPos += windowSize
    
    gradient[:poleEnd] = 0
    gradient[endPos+windowSize:] = 0
    gradient[startPos:endPos-windowSize] = 0

    startPos = np.argmax(gradient)
    endPos = np.argmax(-gradient)
    if endPos < startPos:
        endPos = startPos + 1

    return signalRegion, startPos, endPos
########################################################
########################################################
def plotImage(image, bbox=(360, 300, 373, 430)):
  
    scale_factor = 4
    margin = 50
    image_scaled = image.resize((image.width // scale_factor, image.height // scale_factor))
    bbox_scaled = list(np.array(bbox) * scale_factor)

    image_with_box = addBoundingBoxToImage(image_scaled.copy(), bbox, color=(255, 0, 0), width=5)
    image_cropped = cropImageToBoundingBox(image.copy(), bbox_scaled)
    image_water_gauge, start, end = extractWaterGauge(image.copy(), bbox_scaled)
    image_zoomed = cropImageToBoundingBox(image_cropped, (0, start-margin, image_cropped.width, end+margin))

    print("Start, end positions (in pixels):", start, end)


    fig, axes = plt.subplots(
        1,
        4,
        figsize=(14, 6),
        gridspec_kw={"width_ratios": [4, 1, 1.5, 2]},
    )
    axes[0].imshow(image_with_box)
    axes[0].set_title("Full image")
    axes[0].axis("off")

    axes[1].imshow(image_cropped)
    axes[1].set_title("Cropped")
    axes[1].axhline(start, color='green', linestyle='--', lw=2, label='Start')
    axes[1].axhline(end, color='red', linestyle='--', lw=2, label='End')
    axes[1].axis("off")

    axes[2].set_title("Zoomed")
    axes[2].imshow(image_zoomed)
    axes[2].axhline(margin, color='green', linestyle='--', lw=2, label='Start')
    axes[2].axhline(image_zoomed.height - margin, color='red', linestyle='--', lw=2, label='End')
    axes[2].axis("off")

    axes[3].plot(image_water_gauge, color='blue')
    axes[3].set_title("Intensity profile")
    axes[3].axvline(start, color='green', linestyle='--', lw=2, label='Start')
    axes[3].axvline(end, color='red', linestyle='--', lw=2, label='End')
    axes[3].set_xlabel("Pixels along cropped region")
    axes[3].set_ylabel("Intensity normalized to sidebands")

    plt.subplots_adjust(bottom=0.02, left=0.02, right=0.98, wspace=0.01)

    plt.show()
    return start, end
####################################################################
####################################################################
def loadLabels(jsonFilePath):
    # Resume if labels were already saved earlier.
    if os.path.exists(jsonFilePath):
        with open(jsonFilePath, "r", encoding="utf-8") as f:
            timestamps = {item["timestamp"] for item in json.load(f)}
    else:
        timestamps = set()

    print(f"Already labeled: {len(timestamps)} images")
    return timestamps
######################################################################
######################################################################
def manualSetLabels(images, bbox, timestamps, jsonFilePath):

    stop_requested = False
    labels = []
    for id, image in images:
        if id in timestamps:
            continue

        print(f"\nImage {id} (timestamp: {datetime.fromtimestamp(int(id), tz=timezone.utc).strftime('%Y-%m-%d %H:%M:%S %Z')})")
        plt.close('all')
        start, end = plotImage(image, list(bbox))

        while True:
            user_input = input("Label (value), Enter=skip, s=skip, q=quit-and-save: ").strip()

            if user_input.lower() == "q":
                stop_requested = True
                break

            if user_input == "" or user_input.lower() == "s":
                label_value = None
                skipped = True
            else:
                label_value = int(user_input)
                skipped = False

            labels.append(
                {
                    "label": label_value,
                    "label_ML": str(end),
                    "skipped": skipped,
                    "timestamp": id
                }
            )

            with open(jsonFilePath, "w", encoding="utf-8") as f:
                json.dump(labels, f, indent=2)
            break

        if stop_requested:
            break

    print(f"Done. Total saved records: {len(labels)}")
    print(f"Labels file: {jsonFilePath}")
######################################################################
######################################################################
def runGenAIModelOnImage(image, model="gemini-2.5-flash"):

    # IMPORTANT: Replace 'YOUR_API_KEY' with your actual Google API Key
    # You can get one from https://makersuite.google.com/keys
    # The client gets the API key from the environment variable `GEMINI_API_KEY`.
    client = genai.Client()

    prompt = "Image shows a water gauge in a small river. What is the gauge reading in centimeters?"

    response = client.models.generate_content(
        model=model,
        contents=[client.files.upload(image), prompt],
    )

    ##show the image
    print(response.text)
#########################################################
#########################################################
def plotWaterLevel(df, referencePoint, pixelToCm=1):

    pixelRefPoint, cmRefValue, pixelToCm = referencePoint

    fig, axis = plt.subplots(1, 1, figsize=(8, 4))
    axis.plot(df['timestamp'], (pixelRefPoint -  df['label']) * pixelToCm + cmRefValue, marker='o', linestyle='')
    axis.plot(df['timestamp'], (pixelRefPoint -  df['label_ML']) * pixelToCm + cmRefValue, marker='o', linestyle='')
    axis.set_xlabel('Time (UTC)')
    axis.set_ylabel('Water level [cm]')
    axis.set_ylim(0, 50)
    axis.grid(True)

    axis.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
    axis.tick_params(axis='x', labelrotation=45)

    axis.set_xlim(
        df['timestamp'].min() - pd.Timedelta(days=1),
        df['timestamp'].max() + pd.Timedelta(days=1),
    )
    fig.tight_layout()
##########################################################
##########################################################
def detect_watergauge_bbox(image, verbose=False):
    """
    Detect the rectangular water gauge region in an image.
    
    Args:
        image: PIL Image object or numpy array (grayscale or RGB)
        verbose: if True, print diagnostic info
        
    Returns:
        bbox: tuple (left, top, right, bottom) or None if detection fails
    """
    img_np = np.array(image)
    # Handle PIL images
    #if isinstance(image, Image):
    #    img_np = np.array(image)
    #else:
    #    img_np = image
    
    # Convert to grayscale if needed
    if len(img_np.shape) == 3:
        gray = cv2.cvtColor(img_np.astype(np.uint8), cv2.COLOR_RGB2GRAY)
    else:
        # Already grayscale
        gray = (img_np * 255).astype(np.uint8) if img_np.max() <= 1.0 else img_np.astype(np.uint8)
    
    # Apply edge detection
    edges = cv2.Canny(gray, 50, 150)
    
    # Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        if verbose:
            print("No contours found.")
        return None
    
    # Find the largest rectangle-like contour
    best_bbox = None
    best_area = 0
    
    for contour in contours:
        # Approximate the contour to a polygon
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        
        # Check if it's roughly rectangular (4 corners)
        if len(approx) == 4:
            area = cv2.contourArea(contour)
            print("Area of detected rectangle:", area)
            # Look for a reasonably large rectangle
            if area > best_area and area > 1:
                best_area = area
                x, y, w, h = cv2.boundingRect(contour)
                best_bbox = (x, y, x + w, y + h)
    
    if verbose:
        if best_bbox:
            print(f"Detected water gauge bbox: {best_bbox}, area: {best_area}")
        else:
            print("No rectangular region detected.")
    
    return best_bbox
##########################################################
##########################################################