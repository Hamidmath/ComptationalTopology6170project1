import os
import argparse
import numpy as np
from persim import PersistenceImager
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder

parser = argparse.ArgumentParser(description="Run SVM on Persistence Images")
parser.add_argument("--mode", type=str, choices=["4class", "8class"], default="8class", help="Classification mode")
args = parser.parse_args()


BARCODE_DIR = "barcodes"
OUT_DIR = "results_kernel"
os.makedirs(OUT_DIR, exist_ok=True)

MODE = args.mode
SELECTED_4 = ["cup", "device0", "fish", "flatfish"]

files = sorted([f for f in os.listdir(BARCODE_DIR) if f.endswith("_H1.txt")])
diagrams = []
labels = []

for f in files:
    cls = f.split("-")[0]
    
    if MODE == "4class" and cls not in SELECTED_4:
        continue
   
    dgm = np.loadtxt(os.path.join(BARCODE_DIR, f), ndmin=2)
    diagrams.append(dgm)
    labels.append(cls)

print(f"Using mode: {MODE}")
print("Loaded diagrams:", len(diagrams))

# Convert to persistence images
pimgr = PersistenceImager(
    pixel_size=5,
    birth_range=(0, 100),
    pers_range=(0, 100)
)

pimgr.fit(diagrams)
X = pimgr.transform(diagrams)

# Flatten images
X = np.array([img.flatten() for img in X])

print("Feature matrix shape:", X.shape)


le = LabelEncoder()
y = le.fit_transform(labels)

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.3,
    random_state=42,
    stratify=y
)

clf = SVC(kernel="linear")
clf.fit(X_train, y_train)


y_pred = clf.predict(X_test)

acc = accuracy_score(y_test, y_pred)
print("Accuracy:", acc)
print("\nClassification Report:\n")
report = classification_report(y_test, y_pred, target_names=le.classes_)
print(report)


with open(os.path.join(OUT_DIR, f"svm_persistence_images_results_{MODE}.txt"), "w") as f:
    f.write(f"Accuracy: {acc}\n\n")
    f.write(report)
