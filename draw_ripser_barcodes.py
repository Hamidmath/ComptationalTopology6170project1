import numpy as np
from ripser import ripser
import matplotlib.pyplot as plt
from persim import plot_diagrams
import os

BARCODE_DIR = "barcodes"
BASE_NAME = "Heart-7" 

print(f"Reading barcodes for: {BASE_NAME}")

h0_path = os.path.join(BARCODE_DIR, f"{BASE_NAME}_H0.txt")
h1_path = os.path.join(BARCODE_DIR, f"{BASE_NAME}_H1.txt")

diagrams = []

# Load H0
if os.path.exists(h0_path):
    h0 = np.loadtxt(h0_path)
    if h0.ndim == 1 and h0.size > 0: h0 = h0.reshape(-1, 2)
    if h0.size == 0: h0 = np.empty((0, 2))
    diagrams.append(h0)
    print(f"Loaded H0: {h0.shape}")
else:
    print(f"Warning: {h0_path} not found.")
    diagrams.append(np.empty((0, 2)))

# Load H1
if os.path.exists(h1_path):
    h1 = np.loadtxt(h1_path)
    if h1.ndim == 1 and h1.size > 0: h1 = h1.reshape(-1, 2)
    if h1.size == 0: h1 = np.empty((0, 2))
    diagrams.append(h1)
    print(f"Loaded H1: {h1.shape}")
else:
    print(f"Warning: {h1_path} not found.")
    diagrams.append(np.empty((0, 2)))

plot_diagrams(diagrams, show=False)

# Save the figure to the data directory before showing
output_filename = f"{BASE_NAME}_barcodes.png"
output_path = os.path.join("data", output_filename)
plt.savefig(output_path, bbox_inches='tight')
print(f"Saved barcode image to: {output_path}")

plt.show()
