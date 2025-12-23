# Production_Ready_Credit_Card_Churn_Prediction_System

**End-to-End ML Pipeline | FastAPI | Streamlit | Docker | XGBoost**

---

## 🚀 Project Overview

This project implements a **production-ready, end-to-end customer churn prediction system**, starting from raw data ingestion to real-time predictions via a web interface.

The system covers the **entire machine learning lifecycle**:
- Data ingestion and exploration
- Churn profiling and risk segmentation
- Model comparison and selection
- End-to-end ML pipeline creation
- API-based inference
- UI integration and Dockerized deployment

---

## 🧭 Project Workflow

```
Data Ingestion
      ↓
Data Cleaning & EDA
      ↓
Churn Profiling
      ↓
Risk Segmentation
      ↓
Model Selection
   ↙     ↓      ↘
LogReg  RF   XGBoost
              ↓
       Finalized XGBoost
              ↓
      End-to-End ML Pipeline
              ↓
 FastAPI Inference Service (Pydantic)
              ↓
        Dockerized API
              ↓
   Streamlit UI Integration
              ↓
 Prediction from User Input
```

---

## 🧠 Key Features

- End-to-end **production-grade ML pipeline**
- Custom **sklearn transformers**
- **XGBoost** with categorical feature support
- **Strict request validation** using Pydantic
- Modular, scalable project structure
- Dockerized FastAPI backend
- Interactive Streamlit frontend

---

## 📂 Project Structure

```
churn_prediction_system/
│
├── models/
│   └── xgb_churn_pipeline.pkl
│
├── preprocessing.py
├── app.py
├── streamlit_app.py
│
├── 01_Churn_Profiling_&_Retention_Analysis.ipynb
├── 02_Churn_Prediction.ipynb
│
├── requirements.txt
├── requirements_exact.txt
├── Dockerfile
├── .dockerignore
└── README.md
```

---

## 📊 Notebooks

### 1️⃣ Churn Profiling & Retention Analysis  
**`01_Churn_Profiling_&_Retention_Analysis.ipynb`**

- Customer behavior analysis
- Churn vs non-churn segmentation
- Retention insights and business impact

---

### 2️⃣ Churn Prediction & Modeling  
**`02_Churn_Prediction.ipynb`**

- Feature engineering
- Model training and evaluation
- Model comparison:
  - Logistic Regression
  - Random Forest
  - XGBoost
- Final model selection and pipeline export

---

## 🧩 Machine Learning Pipeline

The saved pipeline (`xgb_churn_pipeline.pkl`) contains:

- Custom preprocessing:
  - Text standardization (lowercasing, trimming)
  - Categorical casting
- Feature transformations
- XGBoost classifier

All preprocessing and modeling logic is **bundled into a single pipeline**, ensuring consistent training and inference behavior.

---

## ⚡ FastAPI Inference Service

### 🔐 Input Validation

- Strong type enforcement
- Range checks
- Categorical value validation
- Automatic error handling

### 🔗 Endpoint

```
POST /predict
```

**Response Example**
```json
{
  "attrition_prediction": 1,
  "attrition_probability": 0.8234
}
```

---

## 🎛 Streamlit Frontend

### Features

- Clean and interactive UI
- User-driven input form
- Real-time API communication
- Clear churn probability visualization

### Communication Flow

```
Streamlit UI → FastAPI → ML Pipeline → Prediction → UI
```

---

## 🐳 Dockerization

The FastAPI inference service is fully containerized.

### Build Docker Image

```bash
docker build -t churn-api .
```

### Run Container

```bash
docker run -p 8000:8000 churn-api
```

---

## ▶️ Running the Project Locally

### 1️⃣ Start FastAPI Server

```bash
uvicorn app:app --reload
```

API available at:
```
http://127.0.0.1:8000
```

---

### 2️⃣ Start Streamlit App

```bash
streamlit run streamlit_app.py
```

UI available at:
```
http://localhost:8501
```

---

## 📦 Tech Stack

- Python
- Pandas / NumPy
- Scikit-learn
- XGBoost
- FastAPI
- Pydantic
- Streamlit
- Docker
- SHAP (optional explainability)

---

## 🎯 Business Value

- Early identification of high-risk churn customers
- Data-driven retention strategy enablement
- Deployable real-time prediction system
- Easily extensible for monitoring and explainability

---

## 🔮 Future Enhancements

- SHAP-based explainability in UI
- Model monitoring and drift detection
- Authentication and access control
- Cloud deployment (AWS / GCP / Azure)

---

## 👤 Author

**Dev Patel**  
Data Science | Machine Learning | Production ML Systems
