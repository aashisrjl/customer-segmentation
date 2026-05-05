# Customer Segmentation & Laptop Purchase Prediction

## Overview

This project performs **dual machine learning tasks**:

1. **Customer Segmentation** using K-Means clustering (5 clusters)
2. **Laptop Purchase Prediction** using Random Forest binary classification

### Key Features
- Real-time predictions via FastAPI REST API
- Interactive web UI form with popup results
- Docker containerization for easy deployment
- Trained ML models ready for production use
- Detailed cluster insights for each prediction

## Quick Start

### Method 1: Local Installation (No Docker)

**Prerequisites**: Python 3.9+

1. **Install dependencies**:
```bash
pip install -r requirements.txt
```

2. **Start the API server**:
```bash
python api.py
```

3. **Access the application**:
   - Open browser: `http://localhost:8000`
   - Form-based predictions with popup results

---

### Method 2: Docker (Recommended)

**Prerequisites**: Docker & Docker Compose installed

1. **Start with Make command**:
```bash
make up
```

2. **Access the application**:
   - Open browser: `http://localhost:8000`

3. **View logs**:
```bash
make logs
```

4. **Stop the application**:
```bash
make down
```

**Available Make commands**:
- `make up` - Start application
- `make down` - Stop application
- `make logs` - View live logs
- `make rebuild` - Rebuild Docker image
- `make clean` - Remove all containers/volumes
- `make help` - Show all commands

---

## Folder Structure

```
customer-segmentation/
├── data/
│   ├── raw/raw_data.csv              # 8,068 customers, 10 features
│   └── processed/
│       ├── processed_data.csv        # Scaled & encoded
│       └── clustered_data.csv        # With cluster labels
├── models/
│   ├── kmeans_model.pkl              # K-Means clustering
│   ├── laptop_classifier.pkl         # Random Forest classifier
│   ├── scaler.pkl                    # Feature scaler
│   └── encoders.pkl                  # Categorical encoders
├── notebooks/
│   ├── 01_EDA.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_clustering.ipynb
│   └── 04_classification.ipynb
├── api.py                            # FastAPI application
├── index.html                        # Web UI form
├── Dockerfile                        # Docker configuration
├── docker-compose.yml                # Docker Compose setup
├── makefile                          # Make commands
├── requirements.txt
└── README.md
```

## REST API

### Predict Endpoint

**GET** `/predict?gender=Male&ever_married=Yes&age=40&graduated=Yes&profession=Engineer&work_experience=5&spending_score=Average&family_size=3`

### Response Example

```json
{
  "status": "success",
  "predicted_cluster": 0,
  "predicted_buy_laptop": "Yes",
  "buy_laptop_probability": 0.96,
  "input_features": {
    "Gender": "Male",
    "Ever_Married": "Yes",
    "Age": "40",
    "Graduated": "Yes",
    "Profession": "Engineer",
    "Work_Experience": "5",
    "Spending_Score": "Average",
    "Family_Size": "3"
  }
}
```

### Customer Clusters Explained

| Cluster | Segment | Key Characteristics | Laptop Purchase |
|---------|---------|-------------------|-----------------|
| **0** | Middle-Aged Professionals | Ages ~47, moderate experience, family-oriented | 32% |
| **1** | Young Singles | Ages ~37, highly experienced, small families | 0% |
| **2** | Young Families | Ages ~26, large families, budget-conscious | 0.4% |
| **3** | General Population | Ages ~46, diverse segment | 27% |
| **4** | Senior Buyers | Ages ~73, smallest segment | **57.5%** ⭐ |

## Model Performance

### Classification (Laptop Purchase Prediction)

| Metric | Value |
|--------|-------|
| F1-Score (5-fold CV) | 0.9988 ± 0.0009 |
| Training Accuracy | 100% |
| Class Distribution | 20% Yes, 80% No |

**Top Features by Importance**:
1. Spending_Score (53.19%)
2. Profession (21.58%)
3. Ever_Married (8.80%)
4. Age (8.38%)
5. Family_Size (4.52%)

### Clustering (K-Means, k=5)

| Metric | Value |
|--------|-------|
| Silhouette Score | 0.2013 |
| Davies-Bouldin Index | 1.7719 |
| Calinski-Harabasz Score | 1549.29 |

## Input Features

| Feature | Type | Allowed Values |
|---------|------|-----------------|
| Gender | Categorical | Male, Female |
| Ever_Married | Categorical | Yes, No |
| Age | Numeric | 18-100 years |
| Graduated | Categorical | Yes, No |
| Profession | Categorical | Artist, Doctor, Engineer, Entertainment, Executive, Healthcare, Homemaker, Lawyer, Marketing |
| Work_Experience | Numeric | 0-60 years |
| Spending_Score | Categorical | Low, Average, High |
| Family_Size | Numeric | 1-10 members |

## Technologies

- **Backend**: FastAPI, Python 3.9
- **ML Libraries**: scikit-learn, pandas, numpy
- **Frontend**: HTML5, CSS3, JavaScript (Vanilla)
- **Deployment**: Docker, Docker Compose
- **Data Processing**: pandas, scikit-learn
