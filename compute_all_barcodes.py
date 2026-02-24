import os
import sys
import argparse
import multiprocessing
import numpy as np
from ripser import ripser
try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

POINTS_DIR = "txtfiles"
OUT_DIR = "barcodes"

os.makedirs(OUT_DIR, exist_ok=True)

files = sorted([f for f in os.listdir(POINTS_DIR) if f.endswith("_points.txt")])

def process_file(f):
    path = os.path.join(POINTS_DIR, f)
    points = np.loadtxt(path)

    result = ripser(points, maxdim=1)
    diagrams = result["dgms"]

    base = f.replace("_points.txt", "")

    np.savetxt(os.path.join(OUT_DIR, base + "_H0.txt"), diagrams[0])
    np.savetxt(os.path.join(OUT_DIR, base + "_H1.txt"), diagrams[1])

    return base

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute barcodes for a subset of files.")
    parser.add_argument("--start", type=int, default=0, help="Start index (0-based)")
    parser.add_argument("--count", type=int, default=None, help="Number of files to process")
    args = parser.parse_args()

    start_index = args.start
    if args.count is not None:
        end_index = start_index + args.count
        files_to_process = files[start_index:end_index]
    else:
        files_to_process = files[start_index:]

    print(f"Processing {len(files_to_process)} files (from index {start_index})")

    if not files_to_process:
        print("No files to process.")
        sys.exit(0)

    num_workers = 2
    print(f"Using {num_workers} cores")

    with multiprocessing.Pool(processes=num_workers, maxtasksperchild=1) as pool:
        if tqdm:
            iterator = tqdm(pool.imap_unordered(process_file, files_to_process), total=len(files_to_process))
        else:
            iterator = pool.imap_unordered(process_file, files_to_process)

        for result in iterator:
            msg = f"COMPLETED: {result}"
            if tqdm:
                tqdm.write(msg)
            else:
                print(msg)

    print("Done.")
