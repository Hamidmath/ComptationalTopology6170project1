import os
import numpy as np
from persim import wasserstein
from joblib import Parallel, delayed

BARCODE_DIR = "barcodes"
OUT_FILE = "data/dist_H1_wasserstein.npy"
OUT_ORDER = "data/H1_wasserstein_file_order.npy"

# Load H1 diagrams
files = sorted([f for f in os.listdir(BARCODE_DIR) if f.endswith("_H1.txt")])
n = len(files)

print("Number of H1 diagrams:", n)

diagrams = []
for f in files:
    dgm = np.loadtxt(os.path.join(BARCODE_DIR, f), ndmin=2)
    diagrams.append(dgm)

dist_matrix = np.zeros((n, n))

def compute_pair(i, j):
    dgm1 = diagrams[i]
    dgm2 = diagrams[j]

    if dgm1.size == 0 or dgm2.size == 0:
        return i, j, 0.0

    d = wasserstein(dgm1, dgm2, matching=False)
    return i, j, d

# Parallel computation
results = Parallel(n_jobs=18, backend="loky", batch_size=20)(
    delayed(compute_pair)(i, j)
    for i in range(n) for j in range(i + 1, n)
)

for i, j, d in results:
    dist_matrix[i, j] = d
    dist_matrix[j, i] = d

np.save(OUT_FILE, dist_matrix)
np.save(OUT_ORDER, np.array(files))

print("Done.")
print("Matrix shape:", dist_matrix.shape)
print("Max distance:", dist_matrix.max())
print("Min distance:", dist_matrix.min())

