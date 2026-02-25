import os
import numpy as np
import pandas as pd

RESULT_DIR = "results_kernel"
OUT_REPORT = os.path.join(RESULT_DIR, "kernel_report.txt")

files = os.listdir(RESULT_DIR)

gram_files = sorted([f for f in files if f.startswith("K_") and f.endswith(".npy")])

print(f"\n===== Stored Kernel Results (Saving to {OUT_REPORT}) =====\n")

with open(OUT_REPORT, "w") as rf:
    rf.write("===== Stored Kernel Results =====\n\n")
    
    for gram_file in gram_files:
        section_header = "--------------------------------------------------\n"
        section_header += f"Kernel Matrix File: {gram_file}\n"
        print(section_header.strip())
        rf.write(section_header)
    
        K = np.load(os.path.join(RESULT_DIR, gram_file))
        
        stats = f"Shape: {K.shape}\n"
        stats += f"Symmetric: {np.allclose(K, K.T)}\n"
        stats += f"Min value: {np.min(K)}\n"
        stats += f"Max value: {np.max(K)}\n"
        stats += f"Mean value: {np.mean(K)}\n"
        
        print(stats.strip())
        rf.write(stats)
    
        mode = gram_file.split("_")[1]  # 4class or 8class
    
        file_path = os.path.join(RESULT_DIR, f"files_{mode}.npy")
        label_path = os.path.join(RESULT_DIR, f"labels_{mode}.npy")
    
        if os.path.exists(file_path) and os.path.exists(label_path):
            filenames = np.load(file_path, allow_pickle=True)
            labels = np.load(label_path, allow_pickle=True)
    
            class_info = f"\nNumber of diagrams: {len(filenames)}\n"
            class_info += f"Unique classes: {sorted(set(labels))}\n"
            class_info += "Class counts:\n"
            class_info += f"{pd.Series(labels).value_counts().to_string()}\n"
            
            print(class_info.strip())
            rf.write(class_info)
    
            preview = "\nFirst 5 entries:\n"
            for i in range(min(5, len(filenames))):
                preview += f"Index {i}: {filenames[i]}  |  Class: {labels[i]}\n"
                
            print(preview.strip())
            rf.write(preview)
    
        footer = "--------------------------------------------------\n\n"
        print(footer.strip())
        rf.write(footer)
    
    rf.write("Done.\n")

print("\nDone.")