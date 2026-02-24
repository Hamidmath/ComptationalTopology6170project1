import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


DIST_PATH = "data/dist_H1_wasserstein.npy"
ORDER_PATH = "data/H1_wasserstein_file_order.npy"

TITLE = "t-SNE — H1 Wasserstein"
OUT_NAME = "tsne_H1_wasserstein.png"


dist_matrix = np.load(DIST_PATH)
files = np.load(ORDER_PATH)

print("Matrix shape:", dist_matrix.shape)

# t-SNE
tsne = TSNE(
    n_components=2,
    metric="precomputed",
    perplexity=20,          # good for n=80
    random_state=42,
    init="random"
)

embedding = tsne.fit_transform(dist_matrix)

print("Embedding shape:", embedding.shape)

# Extract labels
labels = [f.split("-")[0] for f in files]
unique_classes = sorted(set(labels))
class_to_int = {c: i for i, c in enumerate(unique_classes)}
colors = [class_to_int[l] for l in labels]

# Plot
plt.figure(figsize=(8,8))
scatter = plt.scatter(
    embedding[:,0],
    embedding[:,1],
    c=colors,
    cmap="tab10",
    s=40
)

plt.title(TITLE)
plt.xlabel("Component 1")
plt.ylabel("Component 2")

cbar = plt.colorbar(scatter, ticks=range(len(unique_classes)))
cbar.set_ticklabels(unique_classes)

plt.tight_layout()
plt.savefig(f"data/{OUT_NAME}", dpi=300)
plt.show()

print("Saved:", OUT_NAME)
