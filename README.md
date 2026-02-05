# 🎓 Student Social Media Addiction Predictor

A machine learning–based web application that predicts a student's **social media addiction percentage** based on behavioral, academic, and mental health factors.

This project demonstrates an end-to-end ML workflow including data preprocessing, feature engineering, dimensionality reduction, regression modeling, and deployment using Streamlit.

---

## 🚀 Project Overview

Social media addiction has become a growing concern among students, affecting academic performance, mental health, and daily routines.  
This project aims to quantify addiction intensity on a continuous scale and present it as an easy-to-understand **percentage score (0–100%)**.

The system:

- Takes user input related to social media usage and lifestyle
- Processes it through an ML pipeline
- Outputs an addiction score and percentage

---

## ✨ Features

- Outlier Handling using IQR Clipping  
- Categorical Encoding (One-Hot & Ordinal Encoding)  
- Feature Transformations (Logarithmic & Square Root)  
- Dimensionality Reduction using PCA  
- Lasso Regression Model  
- End-to-End Pipeline Integration  
- Streamlit Web Interface  

---

## 📊 Model Performance

The regression model achieves:

**R² Score: 0.9597**

This means the model explains approximately **96% of the variance** in the addiction score.

---

## 🧠 Output Interpretation

- Model predicts a raw addiction score in range **2 – 9**
- Raw score is mapped to **0 – 100%**

Formula: 
percentage = (raw_score - 2) / 7 * 100

Example: 
Raw Score = 6
Percentage = (6 - 2) / 7 * 100 ≈ 57%


---

## ⚙️ Requirements & Dependencies

A `requirements.txt` file is provided as a **best-practice artifact**.

⚠️ Important Notes:

- The file may contain **some unused or extra libraries**
- It was generated mainly to demonstrate good project practice
- The project uses **basic and commonly available libraries**
- The code can run on **most modern versions** of required libraries

You may freely install dependencies manually if preferred.

---

## ▶️ Run Locally

### 1️⃣ (Optional) Create Environment

conda create -n ml_env python=3.10
conda activate ml_env

### 2️⃣ Install Dependencies

pip install -r Deployment/requirements.txt

### 3️⃣ Run Web App

cd Deployment
streamlit run app.py

