from fastapi import FastAPI, Query, HTTPException
from pydantic import BaseModel
from typing import Dict, Union
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import os

from backend.utils import score_sop_with_gemini

app = FastAPI()

BASE_DIR = Path(__file__).resolve().parent
model = joblib.load(BASE_DIR / "models" / "admission_model.pkl")
scaler = joblib.load(BASE_DIR / "models" / "scaler.pkl")

class AdmissionInput(BaseModel):
    gre_score: float
    toefl_score: float
    university_rating: int
    sop: float
    lor: float
    cgpa: float
    research: int

class SOPRequest(BaseModel):
    sop: str

@app.post("/predict_admission")
def predict_admission(input: AdmissionInput) -> Union[Dict[str, float], Dict[str, str]]:
    data = np.array([[
        input.gre_score,
        input.toefl_score,
        input.university_rating,
        input.sop,
        input.lor,
        input.cgpa,
        input.research
    ]])

    scaled_data = scaler.transform(data)
    prediction = model.predict(scaled_data)[0]
    admission_prob = prediction * 100

    if admission_prob < 0:
        return {"message": "Admission unlikely: predicted score is negative."}
    else:
        return {"admission_probability": round(float(admission_prob), 2)}

@app.post("/explain_prediction")
def explain_prediction(input: AdmissionInput):
    from shap import Explainer, maskers

    data = np.array([[
        input.gre_score, input.toefl_score, input.university_rating,
        input.sop, input.lor, input.cgpa, input.research
    ]])

    scaled_data = scaler.transform(data)
    explainer = Explainer(model.predict, masker=maskers.Independent(data=np.zeros((1, 7))))
    shap_values = explainer(scaled_data)

    feature_names = ["GRE Score", "TOEFL Score", "University Rating", "SOP", "LOR", "CGPA", "Research"]
    shap_value_array = shap_values.values[0]
    base_value = shap_values.base_values[0]
    final_prediction = model.predict(scaled_data)[0]

    feature_contributions = dict(zip(feature_names, shap_value_array))

    recommendations = []
    for feature, value in feature_contributions.items():
        if value < -0.05:
            if feature == "GRE Score":
                recommendations.append("Consider retaking the GRE to improve your score.")
            elif feature == "TOEFL Score":
                recommendations.append("Improve your TOEFL score to boost your chances.")
            elif feature == "University Rating":
                recommendations.append("Applying to better-rated universities could help.")
            elif feature == "SOP":
                recommendations.append("Enhance the clarity or relevance of your SOP.")
            elif feature == "LOR":
                recommendations.append("Stronger Letters of Recommendation may help.")
            elif feature == "CGPA":
                recommendations.append("A higher CGPA could significantly improve your profile.")
            elif feature == "Research":
                recommendations.append("Gaining research experience can positively influence your chances.")

    return {
        "base_prediction": round(float(base_value * 100), 2),
        "final_prediction": round(float(final_prediction * 100), 2),
        "shap_values": feature_contributions,
        "recommendations": recommendations or ["Your profile looks strong based on current data."]
    }

@app.get("/lookup")
def lookup_university(name: str = Query(...)):
    df = pd.read_excel(BASE_DIR / "../data/UpdatedWorldUniRank23.xlsx")

    def safe(value):
        if pd.isna(value):
            return "Unavailable"
        if isinstance(value, (np.integer, np.floating)):
            return value.item()
        return str(value)

    result = df[df['University Name'].str.lower() == name.lower()]

    if not result.empty:
        row = result.iloc[0]
        country = safe(row.get("Country"))

        try:
            rank_str = str(row["Rank"])
            if "-" in rank_str:
                low, high = map(int, rank_str.split("-"))
                avg_rank = (low + high) / 2
            else:
                avg_rank = int(rank_str)

            if avg_rank <= 100:
                rating = 5
            elif avg_rank <= 250:
                rating = 4
            elif avg_rank <= 500:
                rating = 3
            else:
                rating = 2

        except Exception:
            rating = 1

        return {
            "University": safe(name),
            "Country": country,
            "Rating (out of 5)": int(rating),
            "Details": {
                "Rank": safe(row.get("Rank")),
                "No of Students": safe(row.get("No of student")),
                "Students per Staff": safe(row.get("No of student per staff")),
                "International Students": safe(row.get("International Student")),
                "Female:Male Ratio": safe(row.get("Female:Male Ratio")),
                "Overall Score": safe(row.get("OverAll Score")),
                "Teaching Score": safe(row.get("Teaching Score")),
                "Research Score": safe(row.get("Research Score")),
                "Citations Score": safe(row.get("Citations Score")),
                "Industry Income Score": safe(row.get("Industry Income Score")),
                "International Outlook Score": safe(row.get("International Outlook Score"))
            }
        }
    else:
        return {
            "University": name,
            "Country": "Unavailable",
            "Rating (out of 5)": 1,
            "Details": {
                "Note": "University not found in database. Assigned default lowest rating."
            }
        }

@app.post("/score_sop")
def score_sop(data: SOPRequest):
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="Missing Gemini API key.")
    return score_sop_with_gemini(data.sop, api_key)
