from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pickle
import numpy as np
import pandas as pd
import shap
from sklearn.preprocessing import LabelEncoder

 #Load saved artifacts 
model     = pickle.load(open('model.pkl',     'rb'))
scaler    = pickle.load(open('scaler.pkl',    'rb'))
threshold = pickle.load(open('threshold.pkl', 'rb'))
features  = pickle.load(open('features.pkl',  'rb'))

#Create SHAP explainer
explainer = shap.TreeExplainer(model)

#Create FastAPI app
app = FastAPI(title="Employee Attrition Predictor")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

#  Define input schema
class Employee(BaseModel):
    Age: int
    BusinessTravel: str
    DailyRate: int
    Department: str
    DistanceFromHome: int
    Education: int
    EducationField: str
    EnvironmentSatisfaction: int
    Gender: str
    HourlyRate: int
    JobInvolvement: int
    JobLevel: int
    JobRole: str
    JobSatisfaction: int
    MaritalStatus: str
    MonthlyIncome: int
    MonthlyRate: int
    NumCompaniesWorked: int
    OverTime: str
    PercentSalaryHike: int
    PerformanceRating: int
    RelationshipSatisfaction: int
    StockOptionLevel: int
    TotalWorkingYears: int
    TrainingTimesLastYear: int
    WorkLifeBalance: int
    YearsAtCompany: int
    YearsInCurrentRole: int
    YearsSinceLastPromotion: int
    YearsWithCurrManager: int

#Encoding maps
encoding_maps = {
    'BusinessTravel':  {'Non-Travel': 0, 'Travel_Frequently': 1, 'Travel_Rarely': 2},
    'Department':      {'Human Resources': 0, 'Research & Development': 1, 'Sales': 2},
    'EducationField':  {'Human Resources': 0, 'Life Sciences': 1, 'Marketing': 2,
                        'Medical': 3, 'Other': 4, 'Technical Degree': 5},
    'Gender':          {'Female': 0, 'Male': 1},
    'JobRole':         {'Healthcare Representative': 0, 'Human Resources': 1,
                        'Laboratory Technician': 2, 'Manager': 3,
                        'Manufacturing Director': 4, 'Research Director': 5,
                        'Research Scientist': 6, 'Sales Executive': 7,
                        'Sales Representative': 8},
    'MaritalStatus':   {'Divorced': 0, 'Married': 1, 'Single': 2},
    'OverTime':        {'No': 0, 'Yes': 1}
}

#Prediction endpoint
@app.post("/predict")
def predict(employee: Employee):

    # Step 1: Convert to dictionary
    data = employee.dict()

    # Step 2: Encode text columns
    for col, mapping in encoding_maps.items():
        data[col] = mapping.get(data[col], 0)

    # Step 3: Create dataframe in correct feature order
    df = pd.DataFrame([data])[features]

    # Step 4: Scale numerical columns
    numerical_columns = df.select_dtypes(include=['int64','float64']).columns.tolist()
    df[numerical_columns] = scaler.transform(df[numerical_columns])

    # Step 5: Get prediction and probability
    probability = model.predict_proba(df)[0][1]
    prediction  = int(probability >= threshold)

    # Step 6: Get SHAP values
    shap_values = explainer.shap_values(df)
    shap_dict   = dict(zip(features, shap_values[0].tolist()))

    # Step 7: Get top 5 factors
    sorted_shap = sorted(shap_dict.items(), key=lambda x: abs(x[1]), reverse=True)[:5]

    top_factors = [
        {
            "feature":   k,
            "impact":    round(v, 3),
            "direction": "increases risk" if v > 0 else "decreases risk"
        }
        for k, v in sorted_shap
    ]

    return {
        "prediction":   "Will Leave" if prediction == 1 else "Will Stay",
        "probability":  round(float(probability), 3),
        "flight_risk":  f"{probability:.1%}",
        "threshold_used": threshold,
        "top_factors":  top_factors
    }

#Health check endpoint
@app.get("/")
def root():
    return {"status": "Employee Attrition API is running!"}