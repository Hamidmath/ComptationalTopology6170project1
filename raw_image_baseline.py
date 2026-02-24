import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.manifold import MDS, TSNE
from sklearn.metrics import pairwise_distances


IMAGE_DIR = "SelectedSubset"

# data process
files = sorted([f for f in os.listdir(IMAGE_DIR) if f.endswith(".gif")])

print("Number of images:", len(files))

images = []
labels = []

for f in files:
    path = os.path.join(IMAGE_DIR, f)

    pil_img = Image.open(path).convert("L")
    pil_img = pil_img.resize((64, 64))
    img = np.array(pil_img)

    # Normalize
    img = img / 255.0

    images.append(img.flatten())
    labels.append(f.split("-")[0])

X = np.array(images)
print("Feature matrix shape:", X.shape)


# Compute Euclidean distance matrix
dist_matrix = pairwise_distances(X, metric="euclidean")

np.save("data/dist_raw_euclidean.npy", dist_matrix)

# MDS
mds = MDS(
    n_components=2,
    metric=True,
    dissimilarity="precomputed",
    random_state=42,
    n_init=4
)

embedding_mds = mds.fit_transform(dist_matrix)

# t-SNE
tsne = TSNE(
    n_components=2,
    metric="precomputed",
    perplexity=20,
    random_state=42,
    init="random"
)

embedding_tsne = tsne.fit_transform(dist_matrix)


# Color
unique_classes = sorted(set(labels))
class_to_int = {c: i for i, c in enumerate(unique_classes)}
colors = [class_to_int[l] for l in labels]


# Plot MDS
plt.figure(figsize=(8,8))
scatter = plt.scatter(
    embedding_mds[:,0],
    embedding_mds[:,1],
    c=colors,
    cmap="tab10",
    s=40
)

plt.title("MDS — Raw Images (Euclidean)")
plt.colorbar(scatter, ticks=range(len(unique_classes)))
plt.tight_layout()
plt.savefig("data/mds_raw_images.png", dpi=300)
plt.show()

# Plot t-SNE
plt.figure(figsize=(8,8))
scatter = plt.scatter(
    embedding_tsne[:,0],
    embedding_tsne[:,1],
    c=colors,
    cmap="tab10",
    s=40
)

plt.title("t-SNE — Raw Images (Euclidean)")
plt.colorbar(scatter, ticks=range(len(unique_classes)))
plt.tight_layout()
plt.savefig("data/tsne_raw_images.png", dpi=300)
plt.show()

print("Done.")
