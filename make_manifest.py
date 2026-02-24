import os

DATA_DIR = "SelectedSubset"
OUT_FILE = "data/selected_files.txt"

classes = ["cup", "Comma", "device0", "device4", "device8", "Heart", "fish", "flatfish"]

lines = []

for fname in sorted(os.listdir(DATA_DIR)):
    if not fname.endswith(".gif"):
        continue
    
    for c in classes:
        if fname.startswith(c + "-"):
            lines.append(f"{c},{os.path.join(DATA_DIR, fname)}")
            break

with open(OUT_FILE, "w") as f:
    f.write("\n".join(lines))

print(f"Wrote {len(lines)} entries to {OUT_FILE}")
