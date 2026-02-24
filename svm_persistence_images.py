import os
import numpy as np
from persim import PersistenceImager
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder


BARCODE_DIR = "barcodes"


# H1 for all classes
files = sorted([f for f in os.listdir(BARCODE_DIR) if f.endswith("_H1.txt")])

diagrams = []
labels = []

for f in files:
    dgm = np.loadtxt(os.path.join(BARCODE_DIR, f), ndmin=2)
    diagrams.append(dgm)
    labels.append(f.split("-")[0])


# H1 for 4-class
# selected_classes = ["cup", "device0", "fish", "flatfish"]

# files = sorted([f for f in os.listdir(BARCODE_DIR) if f.endswith("_H1.txt")])

# diagrams = []
# labels = []

# for f in files:
#     cls = f.split("-")[0]

#     if cls not in selected_classes:
#         continue

#     dgm = np.loadtxt(os.path.join(BARCODE_DIR, f), ndmin=2)
#     diagrams.append(dgm)
#     labels.append(cls)
# print("Using classes:", selected_classes)





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

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred, target_names=le.classes_))



with open("results_kernel/svm_persistence_images_results_8class.txt", "w") as f:
    f.write("Accuracy: " + str(accuracy_score(y_test, y_pred)) + "\n\n")
    f.write(classification_report(y_test, y_pred, target_names=le.classes_))
