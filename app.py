import streamlit as st
import pandas as pd
import joblib
import plotly.graph_objects as go

# Page configuration
st.set_page_config(
    page_title="Breast Cancer Risk Prediction",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for professional medical theme
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .prediction-positive {
        background: linear-gradient(135deg, #ff9a9e 0%, #fecfef 100%);
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #ff4757;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin: 1rem 0;
    }
    .prediction-negative {
        background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%);
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #2ed573;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin: 1rem 0;
    }
    .info-card {
        background: linear-gradient(135deg, #d299c2 0%, #fef9d7 100%);
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .content-text {
        line-height: 1.8;
        font-size: 1.05rem;
    }
</style>
""", unsafe_allow_html=True)

# Load the trained RF model
@st.cache_resource
def load_model():
    try:
        model = joblib.load('rf.pkl')
        return model, None
    except FileNotFoundError:
        return None, "Model file 'rf.pkl' not found. Please ensure the model file is in the correct directory."
    except Exception as e:
        return None, f"Error loading model: {str(e)}"

model, model_error = load_model()
if model_error:
    st.error(f"⚠️ {model_error}")
    st.stop()

# Define all features used in training (exact order from your original df_encoded)
feature_cols = [
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

def prepare_data_for_model(df, model_features):
    df_model = df.copy()
    for col in model_features:
        if col not in df_model.columns:
            df_model[col] = 0
    df_model = df_model[model_features]
    return df_model

# Header
st.markdown("""
<div class="main-header">
    <h1>🏥 Breast Cancer Risk Prediction</h1>
    <p>Advanced Machine Learning for Breast Cancer Prognosis Assessment</p>
</div>
""", unsafe_allow_html=True)

# Tabs for Home & Analyze
tab1, tab2 = st.tabs(["🏠 Home", "📊 Analyze Risk"])

with tab1:
    st.markdown("### Understanding Breast Cancer")
    st.markdown("""
    <div class="info-card content-text">
        Breast cancer is the most common cancer diagnosed in women and the second most common cause of death from cancer among women worldwide. The breasts are paired glands of variable size and density that lie superficial to the pectoralis major muscle. They contain milk-producing cells arranged in lobules; multiple lobules are aggregated into lobes with interspersed fat. Milk and other secretions are produced in acini and extruded through lactiferous ducts that exit at the nipple. Breasts are anchored to the underlying muscular fascia by Cooper ligaments, which support the breast. Breast cancer most commonly arises in the ductal epithelium (ie, ductal carcinoma) but can also develop in the breast lobules (ie, lobular carcinoma). Several risk factors for breast cancer have been well described. In Western countries, screening programs have succeeded in identifying most breast cancers through screening rather than due to symptoms. However, in much of the developing world, a breast mass or abnormal nipple discharge is often the presenting symptom. Breast cancer is diagnosed through physical examination, breast imaging, and tissue biopsy. Treatment options include surgery, chemotherapy, radiation, hormonal therapy, and, more recently, immunotherapy. Factors such as histology, stage, tumor markers, and genetic abnormalities guide individualized treatment decisions.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### Breast Cancer Risk Factors")
    st.markdown("""
    <div class="info-card content-text">
        Identifying factors associated with an increased incidence of breast cancer development is important in general health screening for women. Risk factors for breast cancer include:<br><br>
        <strong>Age:</strong> The age-adjusted incidence of breast cancer continues to increase with the advancing age of the female population.<br><br>
        <strong>Gender:</strong> Most breast cancers occur in women.<br><br>
        <strong>Personal history:</strong> A history of cancer in one breast increases the likelihood of a second primary cancer in the contralateral breast.<br><br>
        <strong>Histologic:</strong> Histologic abnormalities diagnosed by breast biopsy constitute an essential category of breast cancer risk factors. These abnormalities include lobular carcinoma in situ (LCIS) and proliferative changes with atypia.<br><br>
        <strong>Family history and genetic mutations:</strong> First-degree relatives of patients with breast cancer have a 2-fold to 3-fold excess risk for the development of the disease. Genetic factors cause 5% to 10% of all breast cancer cases but may account for 25% of cases in women younger than 30 years. BRCA1 and BRCA2 are the most important genes responsible for increased breast cancer susceptibility.<br><br>
        <strong>Reproductive:</strong> Reproductive milestones that increase a woman’s lifetime estrogen exposure are thought to increase breast cancer risk. These include the onset of menarche before age 12, first live childbirth after age 30 years, nulliparity, and menopause after the age of 55.<br><br>
        <strong>Exogenous hormone use:</strong> Therapeutic or supplemental estrogen and progesterone are taken for various conditions, with the most common scenarios being contraception in premenopausal women and hormone replacement therapy in postmenopausal women.<br><br>
        <strong>Other:</strong> Radiation, environmental exposures, obesity, and excessive alcohol consumption are some other factors that are associated with an increased risk of breast cancer.
    </div>
    """, unsafe_allow_html=True)

with tab2:
    st.markdown("### 👤 Individual Patient Risk Assessment")
    st.markdown("""
    <div class="info-card">
        <h4>🔍 Patient Information</h4>
        <p>Enter the patient's clinical and pathological information below to get a personalized 10-year mortality risk assessment.</p>
        <p><em>Note: This form uses simplified inputs. The encoded dataset contains many more features that will be set to default values.</em></p>
    </div>
    """, unsafe_allow_html=True)

    # Patient input form
    with st.form("patient_form"):
        st.markdown("#### 📝 Basic Demographics & Tumor Characteristics")
        col1, col2, col3 = st.columns(3)

        with col1:
            age = st.slider('👤 Age at Diagnosis', 20, 100, 50, help="Patient's age when diagnosed")
            tumor_size = st.slider('📏 Tumor Size (mm)', 0, 100, 20, help="Maximum tumor diameter in millimeters")

        with col2:
            lymph_nodes = st.slider('🔗 Positive Lymph Nodes', 0, 50, 1, help="Number of lymph nodes with cancer cells")
            mutation_count = st.slider('🧬 Mutation Count', 0, 500, 10, help="Total number of genetic mutations detected")

        with col3:
            npi = st.slider('📊 Nottingham Prognostic Index', 0.0, 10.0, 4.5, help="Combined prognostic score")

        st.markdown("#### 🔬 Treatment & Biomarkers")
        col4, col5, col6 = st.columns(3)

        with col4:
            chemo = st.selectbox(
                '💊 Chemotherapy',
                [0, 1],
                format_func=lambda x: '✅ Yes' if x == 1 else '❌ No',
                help="Whether patient received chemotherapy treatment"
            )

        with col5:
            er_status = st.selectbox(
                '🧪 ER Status',
                [0, 1],
                format_func=lambda x: '🟢 Positive' if x == 1 else '🔴 Negative',
                help="Estrogen receptor status"
            )

        with col6:
            pr_status = st.selectbox(
                '🧪 PR Status',
                [0, 1],
                format_func=lambda x: '🟢 Positive' if x == 1 else '🔴 Negative',
                help="Progesterone receptor status"
            )

        predict_button = st.form_submit_button("🔍 Analyze Patient Risk", type="primary")

    if predict_button:
        input_data = {
            'Age at Diagnosis': age,
            'Lymph nodes examined positive': lymph_nodes,
            'Mutation Count': mutation_count,
            'Nottingham prognostic index': npi,
            'Tumor Size': tumor_size,
            'Chemotherapy': chemo,
            'ER Status': er_status,
            'PR Status': pr_status
        }

        input_df = pd.DataFrame([input_data])
        input_df_prepared = prepare_data_for_model(input_df, feature_cols)

        with st.spinner("🤖 Analyzing patient data..."):
            try:
                prediction = model.predict(input_df_prepared)[0]
                probability = model.predict_proba(input_df_prepared)[0]
            except Exception as e:
                st.error(f"❌ Prediction error: {str(e)}")
                st.stop()

        st.markdown("---")
        st.markdown("### 📋 Risk Assessment Results")

        risk_score = probability[1]

        if prediction == 1:
            st.markdown(f"""
            <div class="prediction-positive">
                <h3>⚠️ HIGH RISK ASSESSMENT</h3>
                <h2>Risk Score: {risk_score:.1%}</h2>
                <p>This patient has a <strong>high predicted risk</strong> for 10-year mortality based on the provided clinical parameters.</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="prediction-negative">
                <h3>✅ LOW RISK ASSESSMENT</h3>
                <h2>Risk Score: {risk_score:.1%}</h2>
                <p>This patient has a <strong>low predicted risk</strong> for 10-year mortality based on the provided clinical parameters.</p>
            </div>
            """, unsafe_allow_html=True)

        # Risk visualization
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=risk_score * 100,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "10-Year Mortality Risk (%)"},
            delta={'reference': 50},
            gauge={
                'axis': {'range': [None, 100]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 25], 'color': "lightgreen"},
                    {'range': [25, 50], 'color': "yellow"},
                    {'range': [50, 75], 'color': "orange"},
                    {'range': [75, 100], 'color': "red"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 90
                }
            }
        ))

        fig_gauge.update_layout(height=400)
        st.plotly_chart(fig_gauge, use_container_width=True)

        # Patient summary
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### 📊 Patient Summary")
            st.write(f"**Age:** {age} years")
            st.write(f"**Tumor Size:** {tumor_size} mm")
            st.write(f"**Positive Lymph Nodes:** {lymph_nodes}")
            st.write(f"**Chemotherapy:** {'Yes' if chemo else 'No'}")

        with col2:
            st.markdown("#### 🔬 Biomarker Profile")
            st.write(f"**ER Status:** {'Positive' if er_status else 'Negative'}")
            st.write(f"**PR Status:** {'Positive' if pr_status else 'Negative'}")
            st.write(f"**Mutation Count:** {mutation_count}")
            st.write(f"**NPI Score:** {npi:.1f}")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; padding: 2rem; background: #f8f9fa; border-radius: 10px; margin-top: 2rem;">
    <h4>🏥 Breast Cancer Risk Prediction</h4>
    <p>⚠️ <strong>Medical Disclaimer:</strong> This tool is for research and educational purposes only. Always consult healthcare professionals for medical decisions.</p>
    <p><em>Powered by Machine Learning & Advanced Analytics</em></p>
</div>
""", unsafe_allow_html=True)
