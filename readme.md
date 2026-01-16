CHUB Heart Disease Risk Prediction System
An Intelligent Clinical Decision Support System (CDSS) for Cardiovascular Care

This project is a complete end-to-end Machine Learning solution developed for the CHUB Referral and Teaching Hospital. It automates the process of predicting heart disease risk levels (0–4) using 13 clinical features, deployed via a Flask REST API with a modern, responsive web interface.

Project Overview
The system provides medical staff with real-time risk stratification to prioritize patient care.

Dataset: 5,000 patient records.

Target Classes: * 0: No Disease (Green)

1: Very Mild (Yellow-Green)

2: Mild (Orange)

3: Severe (Orange-Red)

4: Immediate Danger (Red)

Technology Stack: Python, Flask, Scikit-Learn, Pandas, HTML5/Bootstrap.

📂 Project Structure
Plaintext

heart_disease_prediction/
├── C:\Users\ICT LAB2\ITLML_801_S_A_25RP21043>/                   #
├── data/
│   └── heart_disease_dataset_CHUD_S_A.csv
├── deployment/
│   ├── best_model.pkl 
│   ├── feature_names.pkl        
│   └── class_labels.pkl  
    └── scaler.pkl  
    └── cmodel_info.pkl  

── Training_model.ipynb   
├── index_25RP21043.html          
├── app_25RP21043.py                  
├── requirements.txt        
└── README.md   

 Installation & Setup

1. Create Virtual Environment

python -m ITLML_801_S_A_25RP21043 venv

# Activate (Windows)
ITLML_801_S_A_25RP21043>\Scripts\activate


32. Install Dependencies
pip install -r requirements.txt

Model Training & Evaluation
The model was selected after a rigorous comparison of MLP/ANN, Random Forest, SVM, KNN, and Gradient Boosting using GridSearchCV.

Key Preprocessing Steps:
Imputation: Median for numeric, constant for categorical.

Scaling: StandardScaler for all numerical features.

Encoding: OneHotEncoder for categorical features.

Stratification: Preserved class balance during the 80/20 train-test split.

 API Documentation
Endpoint: /api/predict
Method: POST

Payload Type: application/json

Example Request:

JSON

{
  "age": 63,
  "sex": 1,
  "cp": 3,
  "trestbps": 145,
  "chol": 233,
  "fbs": 1,
  "restecg": 0,
  "thalach": 150,
  "exang": 0,
  "oldpeak": 2.3,
  "slope": 0,
  "ca": 0,
  "thal": 1
}
Example Response:

JSON

{
  "prediction": 4,
  "label": "Immediate Danger",
  "confidence": "94.2%",
  "color": "red"
}

 Web Interface
The system includes a modern Single Page Interface that allows medical staff to:

Enter 13 clinical parameters via a validated form.

View instantaneous risk predictions.

See a visual, color-coded risk meter.

Review per-class probabilities to assist in nuanced clinical diagnosis.

Testing
The system includes robust error handling for:

Data Type Validation: Ensuring numeric inputs are not strings.

Missing Fields: API returns a 400 Bad Request if data is incomplete.

Model Consistency: Verified that the API outputs match the notebook test results.

 License
This project is developed for CHUB Referral and Teaching Hospital. All patient data is anonymized and compliant with healthcare data regulations.