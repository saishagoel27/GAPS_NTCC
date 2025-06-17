import google.generativeai as genai
import re
import json
from typing import Dict, Any

def score_sop_with_gemini(sop: str, api_key: str) -> Dict[str, Any]:
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-1.5-flash")

    prompt = f"""
You are an expert admissions officer. Score the following Statement of Purpose on these 7 metrics (each out of 5):

1. Clarity & Coherence
2. Grammar & Language Quality
3. Purpose & Goal Alignment
4. Motivation & Passion
5. Relevance of Background
6. Research Fit
7. Originality & Insight

Return only valid JSON like:
{{
  "Clarity & Coherence": 4.5,
  "Grammar & Language Quality": 5,
  "Purpose & Goal Alignment": 4,
  ...
}}

SOP:
'''{sop}'''
"""

    try:
        response = model.generate_content([prompt])
        text = response.text

        match = re.search(r"\{.*?\}", text, re.DOTALL)
        if not match:
            raise ValueError("No JSON found in Gemini response.")

        raw_json = json.loads(match.group())
        cleaned = {k.strip(): float(v) for k, v in raw_json.items()}
        avg = round(sum(cleaned.values()) / len(cleaned), 2)

        return {
            "individual_scores": cleaned,
            "average_score": avg
        }

    except Exception as e:
        return {
            "individual_scores": {},
            "average_score": 0,
            "error": f"Gemini error: {e}"
        }
