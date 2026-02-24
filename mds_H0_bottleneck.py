import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import MDS


dist_matrix = np.load("data/dist_H0_bottleneck.npy")
files = np.load("data/H0_bottleneck_file_order.npy")

print("Matrix shape:", dist_matrix.shape)


mds = MDS(
    n_components=2,
    metric=True,
    dissimilarity="precomputed",
    random_state=42,
    n_init=4
)

embedding = mds.fit_transform(dist_matrix)


labels = [f.split("-")[0] for f in files]
unique_classes = sorted(set(labels))
class_to_int = {c: i for i, c in enumerate(unique_classes)}
colors = [class_to_int[l] for l in labels]


plt.figure(figsize=(8,8))

scatter = plt.scatter(
    embedding[:, 0],
    embedding[:, 1],
    c=colors,
    cmap="tab10",
    s=40
)

plt.title("MDS — H0 Bottleneck")
plt.xlabel("Component 1")
plt.ylabel("Component 2")

cbar = plt.colorbar(scatter, ticks=range(len(unique_classes)))
cbar.set_ticklabels(unique_classes)

plt.tight_layout()
plt.savefig("data/mds_H0_bottleneck.png", dpi=300)
plt.show()
