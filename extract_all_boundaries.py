import os
import cv2
import numpy as np
from PIL import Image

MANIFEST = "data/selected_files.txt"
OUT_DIR = "txtfiles"

os.makedirs(OUT_DIR, exist_ok=True)

with open(MANIFEST, "r") as f:
    lines = f.readlines()

for line in lines:
    class_name, image_path = line.strip().split(",")

    print(f"Processing: {image_path}")
    try:
        pil_img = Image.open(image_path).convert('L')
        img = np.array(pil_img)
    except Exception as e:
        print(f"Error: Could not read image {image_path}: {e}")
        continue

    _, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    
    if not contours:
        print(f"Warning: No contours found for {image_path}")
        continue

    points = np.vstack(contours).squeeze()

    base = os.path.basename(image_path).replace(".gif", "_points.txt")
    out_path = os.path.join(OUT_DIR, base)

    if points.size == 0:
        print(f"Warning: Empty points array for {image_path}")
        continue

    np.savetxt(out_path, points, fmt="%d")

print("Done extracting all boundaries.")
