# Hybrid Machine Learning for Dinosaur Analysis

## Project Overview

This project applies both **unsupervised** and **supervised machine learning techniques** to dinosaur paleontology datasets. The objective is to discover meaningful geographical eco-zones from fossil occurrence records and use this information to improve dinosaur diet classification.

The project demonstrates a complete machine learning workflow including:

- Data preprocessing
- Exploratory Data Analysis (EDA)
- Clustering and cluster validation
- Feature engineering
- Regression
- Classification
- Hyperparameter tuning
- Model comparison

---

## Datasets

### 1. Dinosaur Genera Dataset
Contains biological and physical characteristics of dinosaur genera.

Examples of attributes:

- Diet
- Length
- Height
- Weight
- Geological age
- Taxonomic information

### 2. Fossil Occurrence Dataset
Contains fossil discovery records with geographical coordinates.

Examples of attributes:

- Latitude
- Longitude
- Fossil location
- Geological context

---

## Project Structure

```text
project/
│
├── dino_ml_project.ipynb
├── ml_project.pdf
├── README.md
│
├── dataset/
│   ├── dinosaur_genera.csv
│   └── dinosaur_locations.csv
│
└── output/
    ├── phase1_distributions.png
    ├── phase1_heatmap.png
    ├── phase2_elbow.png
    ├── phase2_dendrogram.png
    ├── phase2_kmeans.png
    ├── phase2_validity.png
    ├── phase3_regression.png
    ├── phase3_knn_cm.png
    ├── phase3_tree_plot.png
    ├── phase3_rf_importance.png
    ├── phase4_dt_tuning.png
    ├── phase4_knn_tuning.png
    └── phase6_comparison.png