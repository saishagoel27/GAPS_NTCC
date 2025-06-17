import streamlit as st
import requests
import matplotlib.pyplot as plt
import pandas as pd
from io import StringIO
import base64

st.set_page_config(page_title="🎓 Graduate Admission Predictor", layout="wide")

st.markdown("""
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;700&display=swap" rel="stylesheet">
    <style>
    html, body, [class*="css"]  {
        font-family: 'Inter', sans-serif !important;
    }
    .big-font {
        font-size: 42px !important;
        font-weight: 700;
        color: #2b8cbe;
        letter-spacing: 1px;
    }
    .sub-font {
        font-size: 20px;
        color: #444;
        font-weight: 400;
    }
    .recommendation-box {
        background-color: #f0f2f6;
        padding: 12px;
        border-left: 5px solid #2b8cbe;
        margin-bottom: 10px;
        border-radius: 5px;
        font-size: 16px;
        color: #111;
    }
    .error-msg {
        color: #b91c1c;
        font-weight: 600;
        font-size: 16px;
    }
    </style>
""", unsafe_allow_html=True)

st.sidebar.title("📘 Guide")
st.sidebar.info("""
- Fill your academic scores & SOP
- You can also upload SOP from file
- Click 'Predict' to view:
   - Admission Probability 🎯
   - Feature Contributions 📊
   - SOP Evaluation 📝
   - Smart Recommendations ✅
""")

st.markdown("<div class='big-font'>Graduate Admission Prediction System</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-font'>Analyze your profile, get insights, and maximize your admission chances.</div>", unsafe_allow_html=True)
st.markdown("---")

col1, col2 = st.columns([2, 3])

with col1:
    st.subheader("📄 Applicant Profile")
    gre = st.number_input("GRE Score", min_value=260, max_value=340, value=320)
    toefl = st.number_input("TOEFL Score", min_value=0, max_value=120, value=110)

    st.subheader("🏫 University Info")
    uni_name = st.text_input("Enter University Name to Auto-Fill Rating")
    uni_lookup_result = None
    if st.button("🔍 Lookup University") and uni_name:
        try:
            response = requests.get("http://localhost:8000/lookup", params={"name": uni_name})
            result = response.json()
            if result.get("Rating (out of 5)"):
                uni_lookup_result = result["Rating (out of 5)"]
                st.success(f"Rating for {uni_name}: {uni_lookup_result}")
                st.session_state["uni_rating"] = uni_lookup_result
        except:
            st.error("❌ Could not fetch rating.")

    university_rating = st.number_input("University Rating (1-5)", min_value=1, max_value=5, value=st.session_state.get("uni_rating", 3))

    sop = st.number_input("SOP Strength (1.0 - 5.0)", min_value=1.0, max_value=5.0, step=0.1, value=3.5)
    lor = st.number_input("LOR Strength (1.0 - 5.0)", min_value=1.0, max_value=5.0, step=0.1, value=4.0)
    cgpa = st.number_input("CGPA (out of 10)", min_value=0.0, max_value=10.0, step=0.01, value=8.6)
    research = st.radio("Research Experience", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")

    st.subheader("✍️ Statement of Purpose")
    sop_text = st.text_area("Paste SOP", height=150)
    uploaded_file = st.file_uploader("Or upload SOP file (.txt)", type=['txt'])
    if uploaded_file:
        sop_text = uploaded_file.read().decode("utf-8")
        st.success("SOP uploaded and loaded into text box")

    if st.button("🔍 Predict My Admission Chance"):
        with st.spinner("Running prediction..."):
            input_data = {
                "gre_score": gre,
                "toefl_score": toefl,
                "university_rating": university_rating,
                "sop": sop,
                "lor": lor,
                "cgpa": cgpa,
                "research": research
            }
            try:
                pred = requests.post("http://localhost:8000/predict_admission", json=input_data).json()
                explain = requests.post("http://localhost:8000/explain_prediction", json=input_data).json()
                sop_eval = requests.post("http://localhost:8000/score_sop", json={"sop": sop_text}).json()

                st.session_state.prediction = pred
                st.session_state.explain = explain
                st.session_state.sop_eval = sop_eval
                st.session_state.sop_text = sop_text
                st.session_state.input_data = input_data
                st.rerun()
            except Exception as e:
                st.error(f"❌ API Error: {e}")

with col2:
    if "prediction" in st.session_state:
        st.success(f"🎯 Estimated Admission Probability: {st.session_state.prediction['admission_probability']}%")

        st.subheader("📊 Feature Contributions (SHAP)")
        shap_values = st.session_state.explain.get("shap_values", {})
        sorted_items = sorted(shap_values.items(), key=lambda x: abs(x[1]), reverse=True)

        feat_names = [k for k, _ in sorted_items]
        feat_impacts = [v for _, v in sorted_items]
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.barh(feat_names[::-1], feat_impacts[::-1], color='#2b8cbe')
        ax.set_title("SHAP Impact by Feature")
        ax.set_xlabel("Impact on Prediction")
        st.pyplot(fig)

        st.markdown("""
        <p style='font-size: 15px; color: #444;'>
        Each bar above shows how much a feature increased or decreased your admission chances.<br>
        <b>Positive values help</b>, <b>negative values hurt</b> your probability.
        SHAP (SHapley Additive exPlanations) explains how much each input feature contributed to the final prediction.
        </p>
        """, unsafe_allow_html=True)

        st.subheader("✅ Smart Recommendations")
        for rec in st.session_state.explain.get("recommendations", []):
            st.markdown(f"<div class='recommendation-box'>{rec}</div>", unsafe_allow_html=True)

        st.subheader("📝 SOP Evaluation")
        scores = st.session_state.sop_eval.get("individual_scores", {})
        avg = st.session_state.sop_eval.get("average_score", 0)

        if scores:
            cols = st.columns(2)
            for idx, (k, v) in enumerate(scores.items()):
                cols[idx % 2].metric(label=k, value=f"{v}/5")
            st.markdown(f"### 🧮 Average SOP Score: **{avg} / 5**")
        else:
            st.warning("⚠️ SOP scoring failed or returned no metrics.")
            st.error(st.session_state.sop_eval.get("error", "Unknown error during SOP evaluation."))

        st.subheader("📄 Download Prediction Summary")
        summary = f"""
        🎯 Admission Chance: {st.session_state.prediction['admission_probability']}%

        GRE: {gre} | TOEFL: {toefl} | CGPA: {cgpa}
        SOP: {sop} | LOR: {lor} | Research: {research}
        Univ Rating: {university_rating}

        🔍 SHAP Contributions:
        """
        for feat, val in sorted_items:
            summary += f"- {feat}: {val:+.4f}\n"

        summary += "\n📝 SOP Feedback:\n"
        for rec in st.session_state.explain.get('recommendations', []):
            summary += f"- {rec}\n"

        summary += f"\nAverage SOP Score: {avg}/5"

        b64 = base64.b64encode(summary.encode()).decode()
        href = f'<a href="data:file/txt;base64,{b64}" download="admission_summary.txt">📄 Download TXT Report</a>'
        st.markdown(href, unsafe_allow_html=True)
