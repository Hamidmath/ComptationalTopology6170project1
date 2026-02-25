import os
import argparse
import numpy as np
from persim import bottleneck, wasserstein
from joblib import Parallel, delayed
from tqdm import tqdm

parser = argparse.ArgumentParser(description="Compute Topological Distances in Parallel")
parser.add_argument("--mode", type=str, required=True, choices=["h0b", "h0w", "h1b", "h1w"], 
                    help="Mode mapping: h0b = H0 Bottleneck, h0w = H0 Wasserstein, h1b = H1 Bottleneck, h1w = H1 Wasserstein")
args = parser.parse_args()

# Configure parameters based on the chosen mode
if args.mode == 'h0b':
    dim = 'H0'
    metric = 'bottleneck'
    desc = "Computing H0 Bottleneck"
elif args.mode == 'h0w':
    dim = 'H0'
    metric = 'wasserstein'
    desc = "Computing H0 Wasserstein"
elif args.mode == 'h1b':
    dim = 'H1'
    metric = 'bottleneck'
    desc = "Computing H1 Bottleneck"
elif args.mode == 'h1w':
    dim = 'H1'
    metric = 'wasserstein'
    desc = "Computing H1 Wasserstein"

BARCODE_DIR = "barcodes"
OUT_MATRIX = f"data/dist_{dim}_{metric}.npy"
OUT_ORDER = f"data/{dim}_{metric}_file_order.npy"

# Load diagrams
files = sorted([f for f in os.listdir(BARCODE_DIR) if f.endswith(f"_{dim}.txt")])
n = len(files)

print(f"Number of {dim} diagrams:", n)

diagrams = []
for f in files:
    dgm = np.loadtxt(os.path.join(BARCODE_DIR, f), ndmin=2)
    diagrams.append(dgm)

# We use a memory-mapped array for the distance matrix to avoid RAM exhaustion
dist_matrix = np.zeros((n, n), dtype=np.float32)

def compute_pair(i, j, dgm1, dgm2, selected_metric):
    if dgm1.size == 0 or dgm2.size == 0:
        return i, j, 0.0
    
    if selected_metric == 'bottleneck':
        d = bottleneck(dgm1, dgm2)
    else:  # wasserstein
        d = wasserstein(dgm1, dgm2, matching=False)
        
    return i, j, d

# Generate list of tasks
tasks = [(i, j, diagrams[i], diagrams[j]) for i in range(n) for j in range(i + 1, n)]

print(f"Total pairwise distances to compute: {len(tasks)}")

# Run parallel computation with tqdm progress bar
results = Parallel(n_jobs=28, backend="loky", batch_size=50)(
    delayed(compute_pair)(i, j, dgm1, dgm2, metric)
    for i, j, dgm1, dgm2 in tqdm(tasks, desc=desc)
)

# Fill symmetric matrix
for i, j, d in results:
    dist_matrix[i, j] = d
    dist_matrix[j, i] = d

np.save(OUT_MATRIX, dist_matrix)
np.save(OUT_ORDER, np.array(files))

print("Saved matrix to:", OUT_MATRIX)
print("Saved file order to:", OUT_ORDER)

print("Matrix shape:", dist_matrix.shape)
print("Max distance:", dist_matrix.max())
print("Min distance:", dist_matrix.min())

assert np.allclose(dist_matrix, dist_matrix.T)
assert np.allclose(np.diag(dist_matrix), 0)

print("Matrix is symmetric and valid.")
print("Done.")
