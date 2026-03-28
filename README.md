# Customer Lifetime Value Prediction System

> Full-stack ML system for real-time customer lifetime value prediction and segmentation.

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat&logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-009688?style=flat&logo=fastapi&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?style=flat&logo=docker&logoColor=white)
![MongoDB](https://img.shields.io/badge/MongoDB-Atlas-47A248?style=flat&logo=mongodb&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?style=flat&logo=tensorflow&logoColor=white)

---

## Model Results

| Model | MAE | RMSE | R² |
|-------|-----|------|-----|
| **Random Forest (Best)** ⭐ | **0.0001** | **0.0001** | **1.00** |
| Gradient Boosting | 0.0004 | 0.0005 | 1.00 |
| XGBoost | 0.0010 | 0.0012 | 1.00 |
| Deep Learning (Neural Net) | 0.0130 | 0.0182 | 0.996 |

> **Note:** Models trained on synthetic dataset — high R² reflects controlled data distribution.
> Key metric for business: **11.5% MAE improvement over naive baseline**, processing 50ms per prediction.

**API Response Time:** ~50ms | **Deployment:** Containerized (Docker)

---

## Architecture

```
Customer Input (7 features)
         │
         ▼
┌─────────────────┐
│    FastAPI      │  ← Pydantic validation
│   /predict      │
└────────┬────────┘
         │
         ▼
┌─────────────────┐      ┌──────────────┐
│    Feature      │      │   MongoDB    │
│  Engineering    │ ───► │  Prediction  │
│  (18 features)  │      │   Logging    │
└────────┬────────┘      └──────────────┘
         │
         ▼
┌─────────────────┐
│  Random Forest  │  ← Best performing model
│     Model       │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Post-processing │  ← Unscale + Segment + Confidence
│   + Response    │
└─────────────────┘
```

---

## Features

- **Real-time CLV Prediction** — REST API with ~50ms response time
- **18 Engineered Features** — RFM-based: recency_score, spending_velocity, rfm_combined, etc.
- **4 ML Models Compared** — Random Forest, Gradient Boosting, XGBoost, Deep Learning (Neural Net)
- **Customer Segmentation** — High Value / Medium-High / Medium / At Risk based on predicted CLV
- **MongoDB Logging** — Full prediction audit trail with input features + engineered features
- **Complete CRUD API** — Get, filter by segment, export CSV, delete by ID, clear database
- **Docker Ready** — Containerized for consistent deployment

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| ML Models | Random Forest, XGBoost, Gradient Boosting, TensorFlow/Keras |
| API | FastAPI, Pydantic, Uvicorn |
| Database | MongoDB |
| Data | Pandas, NumPy, Scikit-learn |
| Containerization | Docker, Docker Compose |

---

## Quick Start

### Option 1 — Docker (Recommended)

```bash
git clone https://github.com/shakeabhi7/clv_project.git
cd clv-prediction

cp .env.example .env
# Edit .env with your MongoDB URI if needed

docker-compose up --build

# API available at:  http://localhost:8000
# API docs at:       http://localhost:8000/docs
```

### Option 2 — Local Setup

```bash
git clone https://github.com/shakeabhi7/clv_project.git
cd clv-prediction
pip install -r requirements.txt

cp .env.example .env

uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

---

## API Usage

### Predict CLV
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "age": 35,
    "purchase_frequency": 20,
    "avg_order_value": 150.0,
    "num_orders": 25,
    "customer_lifetime_days": 365,
    "recency": 30,
    "frequency_score": 4
  }'
```

### Sample Response
```json
{
  "predicted_clv": 8500.50,
  "customer_segment": "High Value",
  "comparison_to_average": 11.3,
  "confidence_score": 0.95
}
```

### Other Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | /health | API + model status |
| GET | /stats | Training data statistics |
| GET | /database/stats | Prediction statistics from MongoDB |
| GET | /database/predictions | All predictions (paginated) |
| GET | /database/segment/{segment} | Filter by customer segment |
| POST | /database/export | Export predictions to CSV |
| DELETE | /database/prediction/{id} | Delete specific prediction |

**Interactive docs:** http://localhost:8000/docs

---

## Project Structure

```
clv_project/
├── api/
│   └── main.py                   # FastAPI app + all endpoints
├── backend/
│   ├── database.py               # MongoDB — PredictionDatabase class
│   └── utils.py                  # Feature engineering + post-processing
├── src/
│   ├── model.py                  # Model training pipeline
│   ├── model_comparison.py       # Compare RF, XGB, GBM, Neural Net
│   └── feature_engineering.py   # Feature creation logic
├── models/
│   ├── clv_best_model.pkl        # Best model (Git LFS tracked)
│   ├── clv_model.h5              # Neural network model
│   └── model_scaler.pkl          # Feature scaler
├── data/
│   └── customers_data.csv        # Raw dataset
├── notebooks/
│   ├── eda_visualizations.ipynb  # EDA notebook
│   └── eda_visualization.py      # EDA script
├── output/
│   ├── model_comparison.png      # Model comparison charts
│   ├── model_metrics.txt         # Training metrics
│   └── ...                       # Other analysis outputs
├── dataset_generation.py         # Synthetic data generator
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
├── .env.example
└── .gitignore
```

---

## Engineered Features (18 total)

| Feature | Description |
|---------|-------------|
| total_spending | avg_order_value × num_orders |
| recency_score | 1–5 score based on days since last purchase |
| monetary_score | Spending normalized score |
| rfm_combined | recency + frequency + monetary score |
| spending_velocity | Spending per day of customer lifetime |
| avg_days_between_purchases | customer_lifetime / num_orders |
| frequency_per_month | Purchase frequency normalized per month |
| recency_months | Recency in months |
| + 10 more derived features | Age, frequency, order value variations |

---

## Environment Variables

```env
MONGO_URI=mongodb://localhost:27017/
MODEL_PATH=models/clv_best_model.pkl
REFERENCE_DATA_PATH=cleaned_data/customer_data_rfm.csv
API_PORT=8000
```

---

## Contact

**Abhishek**
- LinkedIn: [linkedin.com/in/shakeabhi](https://linkedin.com/in/shakeabhi)
- GitHub: [github.com/shakeabhi7](https://github.com/shakeabhi7)
- Email: kumarabhishekt7@gmail.com
