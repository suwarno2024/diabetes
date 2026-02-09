# 🩺 DiabetesAI — Interactive Diabetes Prediction Laboratory

A comprehensive, beautifully-designed Streamlit web application for predicting diabetes risk using multiple machine learning models with full evaluation metrics and interactive visualizations.

![Python](https://img.shields.io/badge/Python-3.9+-3776AB?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-FF4B4B?logo=streamlit&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-F7931E?logo=scikit-learn&logoColor=white)

## 🌟 Features

### 🔬 Data Explorer
- Full dataset overview with statistical summaries
- Interactive feature distribution histograms & box plots (by outcome)
- Correlation heatmap with hover details

### 🤖 Model Training & Evaluation
10 machine learning models from simple to complex:

| Model | Type | Complexity |
|---|---|---|
| Logistic Regression | Linear | ⭐ |
| Naive Bayes | Probabilistic | ⭐ |
| K-Nearest Neighbors | Instance-based | ⭐⭐ |
| Decision Tree | Tree-based | ⭐⭐ |
| Random Forest | Ensemble | ⭐⭐⭐ |
| Extra Trees | Ensemble | ⭐⭐⭐ |
| AdaBoost | Ensemble | ⭐⭐⭐ |
| Support Vector Machine | Kernel | ⭐⭐⭐ |
| Gradient Boosting | Ensemble | ⭐⭐⭐⭐ |
| Neural Network (MLP) | Deep Learning | ⭐⭐⭐⭐⭐ |

### 📏 Evaluation Metrics
- **Confusion Matrix** (interactive heatmap)
- **Accuracy, Precision, Recall (Sensitivity), F1-Score**
- **Specificity**
- **ROC Curve & AUC** (Area Under Curve)
- **Precision-Recall Curve**
- **Log Loss**
- **5-Fold Cross-Validation Accuracy**

### 🎯 Prediction
- **Manual input** with risk gauge visualization
- **Batch prediction** via CSV upload with downloadable results
- Feature importance visualization for tree-based models

### 📊 Model Comparison
- Side-by-side metrics table
- Grouped bar chart comparison
- ROC & Precision-Recall curve overlay
- Radar chart for multi-dimensional performance view
- Auto-highlight best model by F1-Score

## 🚀 Quick Start

### Option 1 — Run Locally
```bash
# Clone the repository
git clone https://github.com/<your-username>/diabetes-ai.git
cd diabetes-ai

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

### Option 2 — Deploy on Streamlit Cloud
1. Push this repository to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Click **"New app"** → Select this repo → Set main file to `app.py`
4. Click **Deploy** 🎉

## 📂 Project Structure
```
diabetes-ai/
├── app.py              # Main Streamlit application
├── diabetes.csv        # Pima Indians Diabetes dataset
├── requirements.txt    # Python dependencies
└── README.md           # Documentation
```

## 📊 Dataset

**Pima Indians Diabetes Database** (NIDDK)
- 768 samples, 8 features + 1 target
- Binary classification: Diabetic (1) / Non-Diabetic (0)

| Feature | Description |
|---|---|
| Pregnancies | Number of pregnancies |
| Glucose | Plasma glucose concentration (2h OGTT) |
| BloodPressure | Diastolic blood pressure (mm Hg) |
| SkinThickness | Triceps skin fold thickness (mm) |
| Insulin | 2-hour serum insulin (μU/mL) |
| BMI | Body mass index (kg/m²) |
| DiabetesPedigreeFunction | Diabetes pedigree function |
| Age | Age in years |
| Outcome | 0 = Non-diabetic, 1 = Diabetic |

## 🧹 Data Preprocessing
- Zero values in Glucose, BloodPressure, SkinThickness, Insulin, BMI are replaced with NaN
- Imputed using **median grouped by Outcome** class
- Features standardized with **StandardScaler**

## ⚠️ Disclaimer
This application is for **educational and research purposes only**. It is NOT a medical diagnostic tool. Always consult a qualified healthcare professional for medical advice.

## 📜 License
MIT License — free to use, modify, and distribute.
