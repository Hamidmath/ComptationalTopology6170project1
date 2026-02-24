# Project Pipeline

This document explains the order in which to run the scripts in this project, describing what each script does, what data it reads, and what data it outputs.

**1. Data Preparation**
- `make_manifest.py`
  - **Reads:** Raw image files (`.gif`) from the `SelectedSubset/` directory.
  - **Writes:** A manifest file list to `data/selected_files.txt`.
  - **Description:** Scans the image dataset and creates a consistent list of selected files mapped to their object classes.

**2. Boundary Extraction**
- `extract_all_boundaries.py`
  - **Reads:** The manifest from `data/selected_files.txt` and images from `SelectedSubset/`.
  - **Writes:** `(x, y)` coordinate boundary arrays to the `txtfiles/` directory (`*_points.txt`).
  - **Description:** Binarizes each shape image, extracts the outermost contour boundary, and saves the 2D point cloud.
- `draw_boundary.py`
  - **Reads:** A specific boundary file from `txtfiles/` (e.g., `Heart-7_points.txt`).
  - **Description:** A simple utility to visualize a 2D point cloud boundary of a single shape to verify successful extraction.

**3. Persistent Homology Computation**
- `compute_all_barcodes.py`
  - **Reads:** Boundary coordinate files from `txtfiles/` (`*_points.txt`).
  - **Writes:** Topological barcodes (H0 and H1 components) to the `barcodes/` directory (`*_H0.txt` and `*_H1.txt`).
  - **Description:** Runs the `ripser` library on the point clouds to compute their persistent homology up to dimension 1.
- `draw_ripser_barcodes.py`
  - **Reads:** A set of barcode files for a specific shape from `barcodes/`.
  - **Description:** A visualization utility to plot the H0 and H1 persistence diagrams for a single shape.

**4. Distance Computation**
- `compute_H0_bottleneck_parallel.py` / `compute_H1_bottleneck_parallel.py`
  - **Reads:** H0 or H1 barcodes respectively from the `barcodes/` directory.
  - **Writes:** Bottleneck distance matrices and file orders to the `data/` directory (`dist_H0_bottleneck.npy`, `H0_bottleneck_file_order.npy`, etc.).
  - **Description:** Computes pairwise bottleneck distances between topological barcodes in parallel.
- `compute_H0_wasserstein_parallel.py` / `compute_H1_wasserstein_parallel.py`
  - **Reads:** H0 or H1 barcodes respectively from the `barcodes/` directory.
  - **Writes:** Wasserstein distance matrices and file orders to the `data/` directory.
  - **Description:** Computes pairwise Wasserstein distances between topological barcodes in parallel.

**5. Embedding & Visualization**
- `mds_H0_bottleneck.py` / `mds_H1_bottleneck.py` / `mds_H0_wasserstein.py` / `mds_H1_wasserstein.py`
  - **Reads:** Pre-computed distance matrices and file orders from the `data/` directory.
  - **Writes:** 2D MDS scatter plot images to the `data/` directory (e.g., `mds_H0_bottleneck.png`).
  - **Description:** Applies Multi-Dimensional Scaling (MDS) to layout the shapes based on their topological distances and plots the results colored by class.
- `tsne_from_distance.py`
  - **Reads:** A specific topological distance matrix (e.g., H1 Wasserstein) from the `data/` directory.
  - **Writes:** A 2D t-SNE scatter plot image to the `data/` directory.
  - **Description:** Applies t-SNE for an alternative clustering visualization of the distance matrix.

**6. Baselines & Machine Learning**
- `raw_image_baseline.py`
  - **Reads:** Raw image files from `SelectedSubset/`.
  - **Writes:** Euclidean distance baseline matrix (`data/dist_raw_euclidean.npy`) and baseline MDS/t-SNE plots (`data/mds_raw_images.png`, etc.).
  - **Description:** Evaluates a non-topological baseline by computing Euclidean distances between flattened raw pixels and visualizing the embeddings.
- `svm_persistence_images.py`
  - **Reads:** H1 barcodes from the `barcodes/` directory.
  - **Description:** Converts persistence diagrams into grid-based Persistence Images and trains an SVM classifier to predict shape categories.
- `kernel_svm_persistence_scale_space.py`
  - **Reads:** H1 barcodes from the `barcodes/` directory.
  - **Writes:** Computed Persistence Scale Space kernels, class labels, and file lists to the `results_kernel/` directory.
  - **Description:** Computes topological kernel matrices directly from barcodes and trains an SVM. Saves the kernels for further analysis.
- `kernel_rep.py`
  - **Reads:** Kernel matrices (`K_*.npy`) and label/file data from the `results_kernel/` directory.
  - **Description:** An analysis script that verifies matrix structures, print shapes, min/max values, and class distributions for the generated kernels.