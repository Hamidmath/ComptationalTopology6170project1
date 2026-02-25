import argparse
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import MDS

parser = argparse.ArgumentParser(description="Compute MDS Embeddings from Topological Distances")
parser.add_argument("--mode", type=str, required=True, choices=["h0b", "h0w", "h1b", "h1w"], 
                    help="Mode mapping: h0b = H0 Bottleneck, h0w = H0 Wasserstein, h1b = H1 Bottleneck, h1w = H1 Wasserstein")
args = parser.parse_args()

# Configure parameters based on the chosen mode
if args.mode == 'h0b':
    dim = 'H0'
    metric = 'bottleneck'
    title = "MDS — H0 Bottleneck"
elif args.mode == 'h0w':
    dim = 'H0'
    metric = 'wasserstein'
    title = "MDS — H0 Wasserstein"
elif args.mode == 'h1b':
    dim = 'H1'
    metric = 'bottleneck'
    title = "MDS — H1 Bottleneck"
elif args.mode == 'h1w':
    dim = 'H1'
    metric = 'wasserstein'
    title = "MDS — H1 Wasserstein"

dist_file = f"data/dist_{dim}_{metric}.npy"
order_file = f"data/{dim}_{metric}_file_order.npy"
out_png = f"data/mds_{dim}_{metric}.png"

dist_matrix = np.load(dist_file)
files = np.load(order_file)

print(f"Matrix shape for {args.mode}:", dist_matrix.shape)

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

plt.title(title)
plt.xlabel("Component 1")
plt.ylabel("Component 2")

cbar = plt.colorbar(scatter, ticks=range(len(unique_classes)))
cbar.set_ticklabels(unique_classes)

plt.tight_layout()
plt.savefig(out_png, dpi=300)
# plt.show() # Commented out so it does not block the pipeline script
print(f"Saved {out_png}")
