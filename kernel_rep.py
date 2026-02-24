import os
import numpy as np
import pandas as pd

RESULT_DIR = "results_kernel"

files = os.listdir(RESULT_DIR)

gram_files = sorted([f for f in files if f.startswith("K_") and f.endswith(".npy")])
file_order_files = sorted([f for f in files if f.startswith("files_")])
label_files = sorted([f for f in files if f.startswith("labels_")])

print("\n===== Stored Kernel Results =====\n")

for gram_file in gram_files:
    print("--------------------------------------------------")
    print("Kernel Matrix File:", gram_file)

    K = np.load(os.path.join(RESULT_DIR, gram_file))
    print("Shape:", K.shape)
    print("Symmetric:", np.allclose(K, K.T))
    print("Min value:", np.min(K))
    print("Max value:", np.max(K))
    print("Mean value:", np.mean(K))

    mode = gram_file.split("_")[1]  # 4class or 8class

    file_path = os.path.join(RESULT_DIR, f"files_{mode}.npy")
    label_path = os.path.join(RESULT_DIR, f"labels_{mode}.npy")

    if os.path.exists(file_path) and os.path.exists(label_path):
        filenames = np.load(file_path, allow_pickle=True)
        labels = np.load(label_path, allow_pickle=True)

        print("\nNumber of diagrams:", len(filenames))
        print("Unique classes:", sorted(set(labels)))
        print("Class counts:")
        print(pd.Series(labels).value_counts())

        print("\nFirst 5 entries:")
        for i in range(min(5, len(filenames))):
            print(f"Index {i}: {filenames[i]}  |  Class: {labels[i]}")

    print("--------------------------------------------------\n")

print("Done.")