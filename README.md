# 🦕 Hybrid Machine Learning for Dinosaur Analysis
**Unsupervised Eco-Zone Mapping and Supervised Diet Classification of Mesozoic Dinosaurs**

> **Course:** Machine Learning — Faculté des Sciences Semlalia de Marrakech  
> **Student:** Walid Amjahdi (ID: 2334452)  
> **Advisor:** Prof. Fahd Kalloubi  
> **Academic Year:** 2025–2026

---

## Overview

This project combines **unsupervised** and **supervised** machine learning on real paleontological data. Fossil GPS coordinates are first clustered into geographic eco-zones, then those cluster labels are engineered as features to improve dinosaur diet classification.

The full pipeline covers:
- Exploratory data analysis and KNN-based imputation
- Eco-zone discovery via K-Means++, Agglomerative Clustering, and DBSCAN
- Linear Regression for allometric weight prediction
- Diet classification using KNN, Decision Trees, and Random Forest
- Hyperparameter tuning with GridSearchCV + 5-fold Stratified Cross-Validation
- Final model comparison across Accuracy, Precision, Recall, and F1-Score

---

## Project Structure

```
project/
│
├── dino_ml_project.ipynb          # Main Jupyter notebook (all phases)
├── README.md                      # This file
│
├── dataset/
│   ├── dinosaur_genera.csv        # 1,542 dinosaur genera with physical traits
│   └── dinosaur_locations.csv     # 4,951 fossil occurrence records with GPS coordinates
│
└── output/                        # Auto-generated figures (created by the notebook)
    ├── phase1_distributions.png   # Body length / weight / height histograms
    ├── phase1_heatmap.png         # Correlation matrix + diet class distribution
    ├── phase2_elbow.png           # Elbow method — optimal K selection
    ├── phase2_dendrogram.png      # Ward linkage dendrogram (150-site sample)
    ├── phase2_kmeans.png          # K-Means++ clusters — PCA 2-D projection
    ├── phase2_validity.png        # Silhouette & Davies-Bouldin scores per algorithm
    ├── phase3_regression.png      # Linear Regression: actual vs predicted weight
    ├── phase3_knn_cm.png          # KNN confusion matrix
    ├── phase3_tree_plot.png       # Decision Tree structure (top 3 levels)
    ├── phase3_rf_importance.png   # Random Forest feature importance (Gini)
    ├── phase4_dt_tuning.png       # Decision Tree: max_depth vs CV F1
    ├── phase4_knn_tuning.png      # KNN: n_neighbors vs CV F1 (uniform & distance)
    └── phase6_comparison.png      # Final grouped bar chart — all models vs all metrics
```

---

## Datasets

| Dataset | Source | Records | Key Columns |
|---|---|---|---|
| `dinosaur_genera.csv` | Can Özensoy (Kaggle) | 1,542 | `diet`, `length_m`, `weight_kg`, `height_m`, `locomotion`, `geological_period` |
| `dinosaur_locations.csv` | Smruthiiii (Kaggle) | 4,951 | `lat`, `lng`, `diet`, `length_m`, `max_ma`, `min_ma`, `class` |

---

## Installation

**Python 3.8+** is required. Install all dependencies with:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn scipy jupyter
```

Or using a requirements file:

```bash
pip install -r requirements.txt
```

### `requirements.txt`
```
numpy>=1.24
pandas>=2.0
matplotlib>=3.7
seaborn>=0.12
scikit-learn>=1.3
scipy>=1.10
jupyter>=1.0
```

---

## Running the Notebook

1. Clone or download the repository.
2. Place the two CSV files inside a `dataset/` folder at the project root.
3. Launch Jupyter:

```bash
jupyter notebook dino_ml_project.ipynb
```

4. Run all cells in order (`Kernel → Restart & Run All`).  
   All output figures are saved automatically to the `output/` folder.

> **Reproducibility:** All random operations use `random_state=42`. Results are fully reproducible across runs.

---

## Notebook Phases

| Phase | Description |
|---|---|
| **0 — Imports** | Library setup and global plot style |
| **1 — Preprocessing** | Column selection, label standardization, KNN imputation, StandardScaler, EDA |
| **2 — Clustering** | Elbow method, Dendrogram, K-Means++, Agglomerative, DBSCAN, eco-zone profiling |
| **3 — Supervised Learning** | Linear Regression, KNN, Decision Tree (+ rule extraction), Random Forest |
| **4 — Hyperparameter Tuning** | GridSearchCV for Decision Tree and KNN (macro F1, 5-fold stratified CV) |
| **5 — Full Evaluation** | Confusion matrices and classification reports on held-out test set |
| **6 — Model Comparison** | Grouped bar chart across Accuracy, Precision, Recall, F1 for all models |
| **7 — Conclusion** | Key findings, cluster validity, model rankings, limitations |

---

## Key Results

### Clustering
- **Optimal K = 5** eco-zones identified via the Elbow Method
- K-Means++ achieved the best **Silhouette** and **Davies-Bouldin** scores
- DBSCAN (Haversine, radius = 750 km) correctly isolated remote isolated fossil finds as noise

### Regression
| Metric | Value |
|---|---|
| R² Score | 0.704 |
| MAE | — |
| RMSE | — |

### Classification (Held-Out Test Set, Macro Average)

| Model | Accuracy | Precision | Recall | F1-Score |
|---|---|---|---|---|
| KNN (Tuned, K=5) | 0.885 | 0.939 | 0.679 | 0.855 |
| Decision Tree (Tuned) | 0.883 | 0.928 | 0.679 | 0.855 |
| **Optimized Random Forest** | **0.883** | **0.939** | **0.834** | **0.862** |

**Random Forest** achieves the highest macro F1-score and recall, making it the best overall model. The **Tuned Decision Tree** is preferred when interpretability is required, as its IF-THEN rules can serve as a paleontological identification key.

---

## Methodology Highlights

- **KNN Imputation** over `dropna()` — missing physical traits are estimated from morphologically similar taxa, preserving records that would otherwise be lost.
- **Haversine DBSCAN** — uses true spherical distances (in radians) rather than flat-plane Euclidean distance, producing geographically accurate cluster boundaries.
- **Stratified splits** — all train/test splits and cross-validation folds are stratified to preserve the Herbivore/Carnivore/Omnivore ratio in every fold.
- **Macro F1 scoring** — used throughout GridSearchCV to give equal weight to the minority Omnivore class.
- **Leakage control** — `StandardScaler` is fitted exclusively on the training set and then applied to the test set.

---

## Report

The written report (`ml_project.pdf`) covers all phases with figures embedded. To recompile the PDF:

