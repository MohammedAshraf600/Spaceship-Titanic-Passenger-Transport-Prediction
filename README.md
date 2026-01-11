# 🚀 Spaceship Titanic – Passenger Transport Prediction

## 📌 Project Overview
This project aims to predict whether a passenger was **transported to another dimension** during the *Spaceship Titanic* voyage. The solution applies **exploratory data analysis (EDA)**, **data preprocessing**, and **multiple machine learning models** to evaluate performance and select the best-performing classifier.

The dataset is sourced from the **Kaggle Spaceship Titanic competition** and represents a real-world **binary classification problem**.

---

## 🎯 Objective
Predict the target variable:

**Transported** → `True` or `False`

---

## 📂 Dataset Description

| Feature | Description |
|------|------------|
| PassengerId | Unique passenger ID (dropped) |
| HomePlanet | Planet of origin |
| CryoSleep | Whether the passenger was in cryosleep |
| Cabin | Cabin location (Deck / Number / Side) |
| Destination | Destination planet |
| Age | Passenger age |
| VIP | VIP service status |
| RoomService | Amount spent on room service |
| FoodCourt | Amount spent at food court |
| ShoppingMall | Amount spent at shopping mall |
| Spa | Amount spent at spa |
| VRDeck | Amount spent on VR entertainment |
| Name | Passenger name (dropped) |
| **Transported** | 🎯 Target variable |

---

## 🧹 Data Preprocessing
- Dropped irrelevant columns: `PassengerId`, `Name`
- Removed duplicate rows
- Handled missing values:
  - Categorical features → mode
  - Numerical features → mean
- Outlier detection using scatter plots and box plots
- Label Encoding for categorical features
- Correlation analysis using a heatmap

---

## 📊 Exploratory Data Analysis (EDA)
- Distribution analysis for HomePlanet, CryoSleep, VIP, Destination, and Transported
- Grouped visualizations using age bins
- Feature correlation analysis

---

## 🤖 Machine Learning Models
The following models were trained and evaluated:

- Logistic Regression
- K-Nearest Neighbors (KNN)
- Naive Bayes
- Decision Tree Classifier
- Random Forest Classifier
- Gradient Boosting Classifier

**Train/Test Split:**  
- 80% Training  
- 20% Testing  

---

## 📈 Model Performance Comparison

| Model | Accuracy (%) |
|------|-------------|
| **Gradient Boosting Classifier** | **80.24** |
| Random Forest Classifier | 79.49 |
| Logistic Regression | 78.23 |
| Decision Tree Classifier | 74.02 |
| KNN | 72.87 |
| Naive Bayes | 70.28 |

---

## 📉 ROC Curve & AUC
ROC curves and AUC scores were generated for all models to evaluate classification performance. Ensemble models demonstrated superior discrimination capability.

---

## 🛠️ Tools & Technologies
- Python
- Pandas, NumPy
- Matplotlib, Seaborn
- Scikit-learn
- Jupyter Notebook

---

## 📌 Key Insights
- Ensemble models outperform traditional classifiers
- Spending-related features are strong predictors
- Proper preprocessing improves overall accuracy

---

## 🚀 Future Work
- Hyperparameter tuning
- Feature engineering from `Cabin`
- Advanced models (XGBoost, LightGBM)
- Cross-validation
- Model deployment
