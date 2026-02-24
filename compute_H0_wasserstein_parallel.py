import os
import numpy as np
from persim import wasserstein
from joblib import Parallel, delayed


BARCODE_DIR = "barcodes"
OUT_MATRIX = "data/dist_H0_wasserstein.npy"
OUT_ORDER = "data/H0_wasserstein_file_order.npy"


files = sorted([f for f in os.listdir(BARCODE_DIR) if f.endswith("_H0.txt")])
n = len(files)

print("Number of H0 diagrams:", n)

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


results = Parallel(
    n_jobs=2,              # safe for memory
    backend="loky",
    batch_size=20
)(
    delayed(compute_pair)(i, j)
    for i in range(n) for j in range(i + 1, n)
)

for i, j, d in results:
    dist_matrix[i, j] = d
    dist_matrix[j, i] = d


np.save(OUT_MATRIX, dist_matrix)
np.save(OUT_ORDER, np.array(files))

print("Saved matrix to:", OUT_MATRIX)
print("Matrix shape:", dist_matrix.shape)
print("Max distance:", dist_matrix.max())
print("Min distance:", dist_matrix.min())

assert np.allclose(dist_matrix, dist_matrix.T)
assert np.allclose(np.diag(dist_matrix), 0)

print("Matrix is symmetric and valid.")
print("Done.")
