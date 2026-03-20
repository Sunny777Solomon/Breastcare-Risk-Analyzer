import streamlit as st
import joblib
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import time
from pathlib import Path

st.set_page_config(
    page_title="Breasts they could use your support",
    page_icon="🎗️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ──── Custom CSS with background image ──────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@500;700&family=Inter:wght@300;400;500;600&display=swap');

    body {
        font-family: 'Inter', sans-serif;
        background-image: url('bg.png');  /* ← your uploaded image */
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
        background-color: #fdf2f8;
        color: #4a4a4a;
    }
    h1, h2, h3 {
        font-family: 'Playfair Display', serif;
        color: #db2777;
    }
    .stApp > header {
        background: rgba(253, 242, 248, 0.92);
        backdrop-filter: blur(12px);
    }
    .main-block {
        background: rgba(255, 255, 255, 0.94);
        border-radius: 1.5rem;
        padding: 2rem 2.5rem;
        box-shadow: 0 10px 30px rgba(0,0,0,0.08);
        margin: 1.5rem auto;
        max-width: 1100px;
    }
    .section-title {
        border-bottom: 2px solid #fbcfe8;
        padding-bottom: 0.6rem;
        margin-bottom: 1.5rem;
    }
    .stButton > button {
        background: linear-gradient(135deg, #ec4899, #db2777) !important;
        color: white !important;
        border-radius: 999px !important;
        padding: 1rem 3rem !important;
        font-size: 1.3rem !important;
        transition: all 0.3s;
    }
    .stButton > button:hover {
        transform: translateY(-3px);
        box-shadow: 0 10px 25px rgba(236, 72, 153, 0.4) !important;
    }
    .result-high { color: #dc2626; font-size: 3.5rem; font-weight: bold; }
    .result-low  { color: #059669; font-size: 3.5rem; font-weight: bold; }
    .empty-space-fix { margin-bottom: 0 !important; padding-bottom: 0 !important; }
</style>
""", unsafe_allow_html=True)

# ──── Load model ────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    model_path = Path("rf.pkl")
    if not model_path.exists():
        st.error("rf.pkl not found in the app directory. Please upload it to the GitHub repo root.")
        st.stop()
    try:
        return joblib.load(model_path)
    except Exception as e:
        st.error(f"Model loading failed: {str(e)}")
        st.stop()

model = load_model()

# ──── Full feature list (must match training exactly) ───────────────────────────
MODEL_FEATURES = [
    'Chemotherapy', 'ER status measured by IHC', 'ER Status', 'HER2 status measured by SNP6', 'HER2 Status',
    'Hormone Therapy', 'Inferred Menopausal State', 'Radio Therapy', 'PR Status', 'Tumor Stage_1.0',
    'Tumor Stage_2.0', 'Tumor Stage_3.0', 'Tumor Stage_4.0', 'Cancer Type_Breast Sarcoma',
    'Type of Breast Surgery_Mastectomy', 'Cellularity_Low', 'Cellularity_Moderate',
    'Pam50 + Claudin-low subtype_Her2', 'Pam50 + Claudin-low subtype_LumA',
    'Pam50 + Claudin-low subtype_LumB', 'Pam50 + Claudin-low subtype_NC',
    'Pam50 + Claudin-low subtype_Normal', 'Pam50 + Claudin-low subtype_claudin-low',
    'Cancer Type Detailed_Breast Angiosarcoma', 'Cancer Type Detailed_Breast Invasive Ductal Carcinoma',
    'Cancer Type Detailed_Breast Invasive Lobular Carcinoma', 'Cancer Type Detailed_Breast Invasive Mixed Mucinous Carcinoma',
    'Cancer Type Detailed_Breast Mixed Ductal and Lobular Carcinoma', 'Cancer Type Detailed_Invasive Breast Carcinoma',
    'Cancer Type Detailed_Metaplastic Breast Cancer', 'Neoplasm Histologic Grade_2.0',
    'Neoplasm Histologic Grade_3.0', '3-Gene classifier subtype_ER+/HER2- Low Prolif',
    '3-Gene classifier subtype_ER-/HER2-', '3-Gene classifier subtype_HER2+',
    'Tumor Other Histologic Subtype_Lobular', 'Tumor Other Histologic Subtype_Medullary',
    'Tumor Other Histologic Subtype_Metaplastic', 'Tumor Other Histologic Subtype_Mixed',
    'Tumor Other Histologic Subtype_Mucinous', 'Tumor Other Histologic Subtype_Other',
    'Tumor Other Histologic Subtype_Tubular/ cribriform', 'Primary Tumor Laterality_Right',
    'Integrative Cluster_10', 'Integrative Cluster_2', 'Integrative Cluster_3', 'Integrative Cluster_4ER+',
    'Integrative Cluster_4ER-', 'Integrative Cluster_5', 'Integrative Cluster_6', 'Integrative Cluster_7',
    'Integrative Cluster_8', 'Integrative Cluster_9', 'Age at Diagnosis', 'Lymph nodes examined positive',
    'Mutation Count', 'Nottingham prognostic index', 'Tumor Size'
]

# ──── User input fields ─────────────────────────────────────────────────────────
FEATURES = [
    {"key": "Age_at_Diagnosis", "label": "Age at Diagnosis (years)", "min": 20, "max": 100, "step": 1, "type": "number"},
    {"key": "Tumor_Size", "label": "Tumor Size (mm)", "min": 0, "max": 150, "step": 0.1, "type": "number"},
    {"key": "Lymph_nodes_examined_positive", "label": "Positive Lymph Nodes", "min": 0, "max": 50, "step": 1, "type": "number"},
    {"key": "Mutation_Count", "label": "Mutation Count", "min": 0, "max": 200, "step": 1, "type": "number"},
    {"key": "Nottingham_prognostic_index", "label": "Nottingham Prognostic Index", "min": 1.0, "max": 10.0, "step": 0.01, "type": "number"},
    {"key": "Chemotherapy", "label": "Chemotherapy", "type": "select", "options": {0: "❌ No", 1: "✅ Yes"}},
    {"key": "ER_Status", "label": "ER Status", "type": "select", "options": {0: "🔴 Negative", 1: "🟢 Positive"}},
    {"key": "PR_Status", "label": "PR Status", "type": "select", "options": {0: "🔴 Negative", 1: "🟢 Positive"}},
]

COLUMN_MAP = {
    "Age_at_Diagnosis": "Age at Diagnosis",
    "Tumor_Size": "Tumor Size",
    "Lymph_nodes_examined_positive": "Lymph nodes examined positive",
    "Mutation_Count": "Mutation Count",
    "Nottingham_prognostic_index": "Nottingham prognostic index",
    "Chemotherapy": "Chemotherapy",
    "ER_Status": "ER Status",
    "PR_Status": "PR Status",
}

def prepare_input(user_data):
    df = pd.DataFrame([user_data])
    df = df.rename(columns=COLUMN_MAP)
    for col in MODEL_FEATURES:
        if col not in df.columns:
            df[col] = 0.0
    df = df[MODEL_FEATURES]  # enforce exact order
    return df

# ──── UI ────────────────────────────────────────────────────────────────────────
st.markdown(
    """
    <div style="text-align:center; padding: 3rem 0 1.5rem;">
        <h1 style="font-size: 4.2rem; margin:0;">Breasts they could use your support</h1>
        <p style="font-size: 1.6rem; color: #db2777; margin-top: 0.8rem;">Early awareness saves lives.</p>
    </div>
    """,
    unsafe_allow_html=True
)

tab1, tab2 = st.tabs(["🏠 Home", "📊 Analyze Risk"])

with tab1:
    st.markdown("<div class='main-block'>", unsafe_allow_html=True)

    st.markdown("<h2 class='section-title'>Understanding Breast Cancer</h2>", unsafe_allow_html=True)
    st.markdown("""
    Breast cancer is the most common cancer diagnosed in women and the second most common cause of death from cancer among women worldwide. The breasts are paired glands of variable size and density that lie superficial to the pectoralis major muscle. They contain milk-producing cells arranged in lobules; multiple lobules are aggregated into lobes with interspersed fat. Milk and other secretions are produced in acini and extruded through lactiferous ducts that exit at the nipple. Breasts are anchored to the underlying muscular fascia by Cooper ligaments, which support the breast. Breast cancer most commonly arises in the ductal epithelium (ie, ductal carcinoma) but can also develop in the breast lobules (ie, lobular carcinoma). Several risk factors for breast cancer have been well described. In Western countries, screening programs have succeeded in identifying most breast cancers through screening rather than due to symptoms. However, in much of the developing world, a breast mass or abnormal nipple discharge is often the presenting symptom. Breast cancer is diagnosed through physical examination, breast imaging, and tissue biopsy. Treatment options include surgery, chemotherapy, radiation, hormonal therapy, and, more recently, immunotherapy. Factors such as histology, stage, tumor markers, and genetic abnormalities guide individualized treatment decisions.
    """)

    st.markdown("<h3 class='section-title' style='margin-top:2rem;'>Breast Cancer Risk Factors</h3>", unsafe_allow_html=True)

    st.markdown("""
    Identifying factors associated with an increased incidence of breast cancer development is important in general health screening for women. Risk factors for breast cancer include:

    - **Age**: The age-adjusted incidence of breast cancer continues to increase with the advancing age of the female population.
    - **Gender**: Most breast cancers occur in women.
    - **Personal history**: A history of cancer in one breast increases the likelihood of a second primary cancer in the contralateral breast.
    - **Histologic**: Histologic abnormalities diagnosed by breast biopsy constitute an essential category of breast cancer risk factors. These abnormalities include lobular carcinoma in situ (LCIS) and proliferative changes with atypia.
    - **Family history and genetic mutations**: First-degree relatives of patients with breast cancer have a 2-fold to 3-fold excess risk for the development of the disease. Genetic factors cause 5% to 10% of all breast cancer cases but may account for 25% of cases in women younger than 30 years. BRCA1 and BRCA2 are the most important genes responsible for increased breast cancer susceptibility.
    - **Reproductive**: Reproductive milestones that increase a woman’s lifetime estrogen exposure are thought to increase breast cancer risk. These include the onset of menarche before age 12, first live childbirth after age 30 years, nulliparity, and menopause after the age of 55.
    - **Exogenous hormone use**: Therapeutic or supplemental estrogen and progesterone are taken for various conditions, with the most common scenarios being contraception in premenopausal women and hormone replacement therapy in postmenopausal women.
    - **Other**: Radiation, environmental exposures, obesity, and excessive alcohol consumption are some other factors that are associated with an increased risk of breast cancer.
    """)

    st.markdown("</div>", unsafe_allow_html=True)

with tab2:
    st.markdown("<div class='main-block'>", unsafe_allow_html=True)
    st.markdown("<h2 class='section-title'>10-Year Mortality Risk Assessment</h2>", unsafe_allow_html=True)

    with st.form("patient_form"):
        cols = st.columns(2)
        user_input = {}

        for i, f in enumerate(FEATURES):
            with cols[i % 2]:
                if f["type"] == "select":
                    user_input[f["key"]] = st.selectbox(
                        f["label"],
                        options=list(f["options"].keys()),
                        format_func=lambda x: f["options"][x],
                        key=f["key"]
                    )
                else:
                    step_is_float = isinstance(f["step"], float) or f["step"] != int(f["step"])
                    min_val = float(f["min"]) if step_is_float else int(f["min"])
                    max_val = float(f["max"]) if step_is_float else int(f["max"])
                    default_val = float((f["min"] + f["max"]) / 2) if step_is_float else int((f["min"] + f["max"]) / 2)
                    step_val = f["step"] if step_is_float else int(f["step"])

                    user_input[f["key"]] = st.number_input(
                        f["label"],
                        min_value=min_val,
                        max_value=max_val,
                        value=default_val,
                        step=step_val,
                        key=f["key"]
                    )

        analyze_btn = st.form_submit_button("Analyze Now", type="primary", use_container_width=True)

    if analyze_btn:
        with st.spinner("Analyzing your information..."):
            progress_bar = st.progress(0)
            for pct in range(0, 101, 5):
                time.sleep(0.08)
                progress_bar.progress(pct)

            try:
                input_df = prepare_input(user_input)
                prediction = model.predict(input_df)[0]
                probability = model.predict_proba(input_df)[0][1] * 100
                prob_str = f"{probability:.1f}%"

                st.markdown("---")

                if prediction == 1:
                    st.markdown(f"<p class='result-high'>HIGH 10-Year Mortality Risk</p><p style='font-size:4rem;'>{prob_str}</p>", unsafe_allow_html=True)
                    st.error("""
                    **Please consult a doctor immediately.**  
                    A high predicted risk means early intervention can still make a significant difference.  
                    Schedule an appointment with an oncologist as soon as possible.
                    """)
                else:
                    st.markdown(f"<p class='result-low'>LOW 10-Year Mortality Risk</p><p style='font-size:4rem;'>{prob_str}</p>", unsafe_allow_html=True)
                    st.success("""
                    **Keep it up — you're doing wonderfully.**  
                    This is a very encouraging result. Continue regular screenings and healthy habits.
                    """)

                # Gauge
                fig = go.Figure(go.Indicator(
                    mode="gauge+number+delta",
                    value=probability,
                    domain={'x': [0, 1], 'y': [0, 1]},
                    title={'text': "10-Year Mortality Risk (%)"},
                    delta={'reference': 50},
                    gauge={
                        'axis': {'range': [0, 100]},
                        'bar': {'color': "#db2777"},
                        'steps': [
                            {'range': [0, 25], 'color': "#d1fae5"},
                            {'range': [25, 50], 'color': "#fef3c7"},
                            {'range': [50, 75], 'color': "#fed7aa"},
                            {'range': [75, 100], 'color': "#fecaca"}
                        ],
                        'threshold': {'line': {'color': "#dc2626", 'width': 4}, 'thickness': 0.75, 'value': 90}
                    }
                ))
                fig.update_layout(height=350, margin=dict(l=20, r=20, t=50, b=20))
                st.plotly_chart(fig, use_container_width=True)

            except Exception as e:
                st.error(f"Prediction failed: {str(e)}")
                st.info("Possible causes:\n- rf.pkl not found or corrupted\n- Feature mismatch between model and input\n- Missing columns in input data")

    st.markdown("</div>", unsafe_allow_html=True)

# Footer
st.markdown("""
<div style="text-align:center; padding:2rem 1rem; color:#6b7280; font-size:0.95rem;">
    🎗️ Educational awareness tool only — not medical advice. Always consult a healthcare professional.
</div>
""", unsafe_allow_html=True)
