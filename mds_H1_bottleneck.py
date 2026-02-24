import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import MDS

DIST_PATH = "data/dist_H1_bottleneck.npy"
BARCODE_DIR = "barcodes"

dist_matrix = np.load(DIST_PATH)
print("Matrix shape:", dist_matrix.shape)

mds = MDS(
    n_components=2,
    metric=True,
    dissimilarity="precomputed",
    random_state=42,
    n_init=4
)

embedding = mds.fit_transform(dist_matrix)
print("Embedding shape:", embedding.shape)

files = sorted([f for f in os.listdir(BARCODE_DIR) if f.endswith("_H1.txt")])
labels = [f.split("-")[0] for f in files]

unique_classes = sorted(list(set(labels)))
class_to_int = {c: i for i, c in enumerate(unique_classes)}
colors = [class_to_int[label] for label in labels]

print("Classes:", unique_classes)

plt.figure(figsize=(8, 8))

scatter = plt.scatter(
    embedding[:, 0],
    embedding[:, 1],
    c=colors,
    cmap="tab10",
    s=40
)

plt.title("MDS — H1 Bottleneck (Colored by Class)")
plt.xlabel("Component 1")
plt.ylabel("Component 2")

cbar = plt.colorbar(scatter, ticks=range(len(unique_classes)))
cbar.set_ticklabels(unique_classes)

plt.tight_layout()
plt.savefig("data/mds_H1_bottleneck.png", dpi=300)
plt.show()
