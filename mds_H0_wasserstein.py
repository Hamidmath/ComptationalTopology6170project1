import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import MDS

# Paths
DIST_PATH = "data/dist_H0_wasserstein.npy"
ORDER_PATH = "data/H0_wasserstein_file_order.npy"
BARCODE_DIR = "barcodes"

# Load
dist_matrix = np.load(DIST_PATH)
files = np.load(ORDER_PATH)

print("Matrix shape:", dist_matrix.shape)

# MDS
mds = MDS(
    n_components=2,
    metric=True,
    dissimilarity="precomputed",
    random_state=42,
    n_init=4
)
embedding = mds.fit_transform(dist_matrix)
print("Embedding shape:", embedding.shape)

# Labels
labels = [f.split("-")[0] for f in files]
unique_classes = sorted(set(labels))
class_to_int = {c: i for i, c in enumerate(unique_classes)}
colors = [class_to_int[l] for l in labels]

# Plot
plt.figure(figsize=(8,8))
scatter = plt.scatter(embedding[:,0], embedding[:,1], c=colors, cmap="tab10", s=40)
plt.title("MDS — H0 Wasserstein (Colored by Class)")
plt.xlabel("Component 1")
plt.ylabel("Component 2")
cbar = plt.colorbar(scatter, ticks=range(len(unique_classes)))
cbar.set_ticklabels(unique_classes)
plt.tight_layout()
out_png = "data/mds_H0_wasserstein.png"
plt.savefig(out_png, dpi=300)
plt.show()
print("Saved plot to:", out_png)
