import os
import numpy as np
import matplotlib.pyplot as plt

# Directory containing the text files
TXT_DIR = "txtfiles"

FILE_NAME = "Heart-7_points.txt" 

FULL_PATH = os.path.join(TXT_DIR, FILE_NAME)

print(f"Reading file: {FULL_PATH}")

try:
    if not os.path.exists(FULL_PATH):
        raise FileNotFoundError(f"File not found: {FULL_PATH}")

    points = np.loadtxt(FULL_PATH)
    
    if points.size == 0:
        print(f"Warning: Empty file {FULL_PATH}")
    else:
        if len(points.shape) == 1:
             points = points.reshape(-1, 2)

        print(f"Number of boundary points: {points.shape[0]}")

        plt.figure(figsize=(6, 6))
        plt.scatter(points[:, 0], -points[:, 1], s=1, label=FILE_NAME)
        plt.title(f"Boundary Point Cloud: {FILE_NAME}")
        plt.axis('equal')
        plt.legend()
        plt.show()

except Exception as e:
    print(f"Error: {e}")
