# 🚢 Titanic Survival Prediction - ML Pipeline

## 📌 Project Overview
This project implements a complete end-to-end Machine Learning pipeline to predict passenger survival on the Titanic dataset. It demonstrates structured data preprocessing, feature engineering, model training, hyperparameter tuning, and evaluation using industry-standard best practices.

The objective is to build a reproducible and production-style ML workflow rather than just training a simple classification model.

---

## 🧠 Problem Statement
Given passenger information such as age, gender, ticket class, and family details, predict whether a passenger survived the Titanic disaster.

This is a binary classification problem:
- 0 → Did Not Survive  
- 1 → Survived  

---

## ⚙️ Machine Learning Pipeline

### 🔹 Data Preprocessing
- Handling missing values  
- Encoding categorical variables  
- Feature scaling  
- ColumnTransformer for structured preprocessing  

### 🔹 Feature Engineering
- Created **FamilySize** feature using:
  - SibSp (siblings/spouses)
  - Parch (parents/children)

### 🔹 Model Training
- Stratified Train-Test Split  
- 5-Fold Stratified Cross Validation  
- Hyperparameter tuning using GridSearchCV  
- Best model selection  

---

## 📊 Model Performance

- Final Test Accuracy: **0.82 (82%)**
- Cross-Validation Accuracy: ~0.80 – 0.83
- Stable generalization performance
- All test cases executed successfully

Example Output:

Test Accuracy: 0.82  

Confusion Matrix:  
[[90 15]  
 [18 56]]  

<img width="640" height="480" alt="Figure_1" src="https://github.com/user-attachments/assets/96831e84-d4f1-4c2e-ab93-00baf5823210" />


---

## ▶️ How to Run the Project

### 1️⃣ Clone the Repository
git clone https://github.com/saadiftkhar/titanic-survival-ml.git  
cd titanic-survival-ml  

### 2️⃣ Install Dependencies
pip install -r requirements.txt  

### 3️⃣ Run Training Script
python src/train.py  

The script will:
- Train the model  
- Perform cross-validation  
- Evaluate on test data  
- Print performance metrics 

---

### 3️⃣ Run Test Cases
python tests/preprocessing_test.py

The script will:
- Run all test cases    
- Print "All preprocessing tests passed!"

---
  
## 🛠️ Tech Stack
- Python  
- Pandas  
- NumPy  
- Scikit-learn  
- Joblib
- Matplotlib
- Seaborn 

---

## 🎯 Key Learning Outcomes
- Building production-style ML pipelines  
- Structured preprocessing using ColumnTransformer  
- Hyperparameter tuning with GridSearchCV  
- Avoiding data leakage  
- Writing reproducible ML code  

---

## 👨‍💻 Author
Saad Iftikhar
