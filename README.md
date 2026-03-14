Dimensionality Reduction and Classification from ScratchThis project implements Principal Component Analysis (PCA) and Naive Bayes from scratch using Python and NumPy. The goal is to compare the performance of baseline models against Feature Selection and Feature Reduction (PCA) across two distinct datasets.
🚀 Key Features
        PCA from Scratch: Implements mean centering, covariance matrix calculation, eigendecomposition, and data projection.
        Naive Bayes from Scratch: Supports both Gaussian (numerical) and Bernoulli (binary/categorical) distributions using log-likelihood summation.
        Comparative Analysis: Evaluates the trade-offs between full feature sets, manual feature selection, and PCA-reduced features.

📊 Datasets Used
      1- Social Media Addiction Dataset: A mixed dataset containing numerical usage data and categorical demographic labels to predict addiction levels.
      2- Breast Cancer Wisconsin Diagnostic: A high-dimensional medical dataset (30 features) used to identify malignant vs. benign tumors.

🧪 Experiments Performed
     - Experiment 0 (Baseline): Training Naive Bayes on the full original feature set.
     - Experiment A (Feature Selection): Training on a manually selected subset of high-impact features.
     - Experiment B (PCA): Reducing dimensionality to k components and optimizing accuracy.
     
Approach,Social Media Accuracy,Breast Cancer Accuracy

  - Baseline,99.29%,96.49%
  - Feature Selection,99.29%,92.98%
  - PCA (Scratch),97.87%,98.25%

Key Observations
    - PCA Efficiency: In the medical dataset, PCA outperformed the baseline, proving it can filter noise while preserving core information.
    - Feature Selection: For behavioral data, specific features (like "Conflicts") are often as powerful as the entire dataset.

🛠️ Requirements
      - Python 3.x
      - NumPy
      - Pandas
      - Matplotlib
      - Scikit-learn (used only for data loading and metrics)
