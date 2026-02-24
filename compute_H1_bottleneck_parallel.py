import os
import numpy as np
from persim import bottleneck
from concurrent.futures import ProcessPoolExecutor
import multiprocessing

BARCODE_DIR = "barcodes"
OUT_FILE = "data/dist_H1_bottleneck.npy"
OUT_ORDER = "data/H1_bottleneck_file_order.npy"

# Load H1 diagrams
files = sorted([f for f in os.listdir(BARCODE_DIR) if f.endswith("_H1.txt")])
n = len(files)

print("Number of H1 diagrams:", n)

diagrams = []
for f in files:
    dgm = np.loadtxt(os.path.join(BARCODE_DIR, f), ndmin=2)
    diagrams.append(dgm)

# Function to compute one pair distance
def compute_pair(pair):
    i, j = pair
    dgm1 = diagrams[i]
    dgm2 = diagrams[j]

    if dgm1.size == 0 or dgm2.size == 0:
        return (i, j, 0.0)

    d = bottleneck(dgm1, dgm2)
    return (i, j, d)

if __name__ == "__main__":

    # limit cores (avoid memory explosion)
    num_workers = 18
    print("Using", num_workers, "cores")

    # Generate index pairs
    pairs = [(i, j) for i in range(n) for j in range(i+1, n)]

    dist_matrix = np.zeros((n, n))

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        for i, j, d in executor.map(compute_pair, pairs):
            dist_matrix[i, j] = d
            dist_matrix[j, i] = d

    # Save matrix
    np.save(OUT_FILE, dist_matrix)
    np.save(OUT_ORDER, np.array(files))

    print("Done.")
    print("Matrix shape:", dist_matrix.shape)
    print("Max distance:", dist_matrix.max())
    print("Min distance:", dist_matrix.min())
