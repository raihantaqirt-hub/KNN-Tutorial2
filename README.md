# KNN-Tutorial2
## Disclaimer: This repository is a dummy to try my first Git profile.
Implementation of the K-Nearest Neighbors (KNN) algorithm for classification and regression using Python.
# Titanic Survival Prediction using K-Nearest Neighbors (KNN)

An interactive machine learning tutorial demonstrating how to build, evaluate, and optimize a **K-Nearest Neighbors (KNN)** classification model to predict passenger survival using the iconic **Titanic dataset** from Seaborn.

---

## 📌 Project Overview
The goal of this tutorial is to predict whether a passenger survived (`1`) or perished (`0`) during the Titanic disaster. 

Since KNN is a **distance-based algorithm**, this project places heavy emphasis on essential preprocessing steps like **imputing missing values**, **categorical encoding**, and **feature scaling** (StandardScaler), which directly impact KNN performance.

## 🛠️ Tech Stack & Libraries
* **Language:** Python 3.x
* **Libraries:**
  * `seaborn` & `pandas` (Data loading and manipulation)
  * `scikit-learn` (Model building, preprocessing, and evaluation)

---

## 🚀 Getting Started

### 1. Prerequisites
Make sure you have Python installed, then install the required dependencies:

```bash
pip install seaborn pandas scikit-learn
```

### 2. Run the Tutorial
Clone this repository and run the script or Jupyter Notebook:

```bash
git clone https://github.com
cd YOUR_REPO_NAME
python knn_titanic.py
```

---

## 📈 Model Workflow & Key Steps

1. **Data Loading:** Automatically fetches the live dataset using `sns.load_dataset('titanic')`.
2. **Data Cleaning:** 
   * Removes redundant/highly missing columns (`deck`, `alive`, `class`).
   * Handles missing values by filling numerical data with the **median** and categorical data with the **mode**.
3. **Categorical Encoding:** Converts text categories (`sex`, `embarked`) into numerical flags using binary mapping and dummy variables (`pd.get_dummies`).
4. **Feature Scaling (Crucial for KNN):** Normalizes features using `StandardScaler` so that high-magnitude columns (like `fare`) do not artificially skew the geometric distance calculations.
5. **Training:** Fits a `KNeighborsClassifier` (default k=5) to the 80% split training set.
6. **Evaluation:** Measures performance on the remaining 20% test split.

---

## 📊 Sample Output & Evaluation

When you run the tutorial, you can expect an accuracy of roughly **79% - 82%**. The script outputs standard evaluation metrics:

```text
Accuracy Score: 0.8101

Classification Report:
              precision    recall  f1-score   support

           0       0.82      0.87      0.84       105
           1       0.79      0.73      0.76        74

    accuracy                           0.81       179
```

---

## 🧠 Things to Explore in This Tutorial
To maximize your learning from this repo, try modifying the code to explore:
* **Changing the K-Value:** Experiment with different `n_neighbors` (e.g., $k=1, 3, 7, 15$) to see how it affects underfitting and overfitting.
* **Distance Metrics:** Switch the metric parameter from `minkowski` (Euclidean) to `manhattan` inside the classifier configuration.

## 📄 License
This project is licensed under the MIT License - see the LICENSE file for details.
