import os
import argparse
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import time

parser = argparse.ArgumentParser(description="Run Kernel SVM")
parser.add_argument("--mode", type=str, choices=["4class", "8class"], default="8class", help="Classification mode")
args = parser.parse_args()

BARCODE_DIR = "barcodes"   # must contain *_H1.txt
DIAG_SUFFIX = "_H1.txt"
OUT_DIR = "results_kernel"
os.makedirs(OUT_DIR, exist_ok=True)

# Choose mode: "8class" or "4class"
MODE = args.mode   # takes value from command line argument
SELECTED_4 = ["cup", "device0", "fish", "flatfish"]  # used if MODE=="4class"

# Kernel parameter (tuneable)
SIGMA = 100.0

# Random seed for splits
RANDOM_STATE = 42
TEST_SIZE = 0.3


# Utility: load diagrams and labels
files = sorted([f for f in os.listdir(BARCODE_DIR) if f.endswith(DIAG_SUFFIX)])
diagrams = []
labels = []
filenames = []

for f in files:
    cls = f.split("-")[0]
    if MODE == "4class" and cls not in SELECTED_4:
        continue
    dgm = np.loadtxt(os.path.join(BARCODE_DIR, f), ndmin=2)
    diagrams.append(dgm)
    labels.append(cls)
    filenames.append(f)

n = len(diagrams)
print("Loaded", n, "diagrams for mode", MODE)

# Kernel: persistence scale-space
def reflect_diag(d):
    # reflect a single point [b,d] -> [d,b]
    return d[..., ::-1]

def pairwise_kernel_between(diagA, diagB, sigma):
    # diagA: (m,2), diagB: (n,2)
    if diagA.size == 0 or diagB.size == 0:
        return 0.0
    A = diagA[:, None, :]            # shape (m,1,2)
    B = diagB[None, :, :]            # shape (1,n,2)
    diff = A - B                     # shape (m,n,2)
    sq = np.sum(diff * diff, axis=2) # (m,n)
    term1 = np.exp(-sq / (8.0 * sigma))
    # reflect B across diagonal
    B_ref = B[..., ::-1]
    diff2 = A - B_ref
    sq2 = np.sum(diff2 * diff2, axis=2)
    term2 = np.exp(-sq2 / (8.0 * sigma))
    val = np.sum(term1 - term2)
    const = 1.0 / (8.0 * np.pi * sigma)
    return const * val

def compute_gram(diags, sigma, verbose=True):
    n = len(diags)
    K = np.zeros((n, n), dtype=float)
    t0 = time.time()
    for i in range(n):
        # compute j>=i and mirror
        for j in range(i, n):
            kij = pairwise_kernel_between(diags[i], diags[j], sigma)
            K[i, j] = kij
            K[j, i] = kij
        if verbose and (i % 10 == 0 or i == n-1):
            print(f"[{i+1}/{n}] elapsed {time.time()-t0:.1f}s")
    return K

# Build kernel matrix
print("Computing Gram matrix with sigma=", SIGMA)
t0 = time.time()
K = compute_gram(diagrams, SIGMA, verbose=True)
print("Gram computed in {:.1f}s".format(time.time()-t0))

# Save gram and filenames
np.save(os.path.join(OUT_DIR, f"K_{MODE}_sigma{int(SIGMA)}.npy"), K)
np.save(os.path.join(OUT_DIR, f"files_{MODE}.npy"), np.array(filenames))
np.save(os.path.join(OUT_DIR, f"labels_{MODE}.npy"), np.array(labels))


# Train/test split using indices (precomputed kernel)
le = LabelEncoder()
y_all = le.fit_transform(labels)

idx = np.arange(n)
train_idx, test_idx = train_test_split(idx, test_size=TEST_SIZE, stratify=y_all, random_state=RANDOM_STATE)

K_train = K[np.ix_(train_idx, train_idx)]
K_test = K[np.ix_(test_idx, train_idx)]

y_train = y_all[train_idx]
y_test = y_all[test_idx]

print("Train size:", len(train_idx), "Test size:", len(test_idx))


# SVM with precomputed kernel
clf = SVC(kernel="precomputed", C=1.0)
clf.fit(K_train, y_train)

y_pred = clf.predict(K_test)
acc = accuracy_score(y_test, y_pred)
print(f"Accuracy ({MODE}, sigma={SIGMA}): {acc:.4f}")


report = classification_report(y_test, y_pred, target_names=le.classes_)
cm = confusion_matrix(y_test, y_pred)


with open(os.path.join(OUT_DIR, f"kernel_svm_report_{MODE}_sigma{int(SIGMA)}.txt"), "w") as fh:
    fh.write(f"Accuracy: {acc}\n\n")
    fh.write("Classification report:\n")
    fh.write(report)
    fh.write("\nConfusion matrix:\n")
    np.savetxt(fh, cm, fmt="%d")

print("Saved report to results_kernel/")
print(report)
