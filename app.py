import io
import pickle
import streamlit as st
import pandas as pd
from pipeline import train_models
from explainer import get_shap_plot
from reporter import generate_pdf

st.set_page_config(
    page_title="AutoML Pipeline",
    page_icon="⚡",
    layout="wide"
)

st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #0d1b2a, #1a2f4a, #0d2137);
        color: #f8fafc;
    }

    .block-container {
        max-width: 1100px;
        padding-top: 2rem;
    }

    .main-title {
        font-size: 3rem;
        font-weight: 800;
        text-align: center;
        background: linear-gradient(90deg, #00c9a7, #0072ff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    .subtitle {
        text-align: center;
        color: #d1d5db;
        font-size: 1.05rem;
        margin-bottom: 2rem;
    }

    .card {
        background: rgba(255,255,255,0.05);
        border: 1px solid rgba(255,255,255,0.10);
        border-radius: 16px;
        padding: 1.5rem;
        margin-bottom: 1rem;
    }

    .stButton > button {
        background: linear-gradient(90deg, #00c9a7, #0072ff);
        color: white;
        border: none;
        border-radius: 10px;
        font-weight: 700;
        width: 100%;
    }

    .stButton > button:hover {
        opacity: 0.92;
    }

    /* Hide Streamlit default top bar */
    header {
        visibility: hidden;
        height: 0px;
    }

    div[data-testid="stDecoration"] {
        display: none !important;
    }

    /* General text */
    p, label, div, span {
        color: #f8fafc;
    }

    /* File uploader label */
    section[data-testid="stFileUploader"] label,
    section[data-testid="stFileUploader"] > div {
        color: #f8fafc !important;
    }

    /* Drag and drop box text */
    [data-testid="stFileUploaderDropzone"] {
        background: rgba(255,255,255,0.06);
        border: 1px dashed rgba(255,255,255,0.25);
        border-radius: 14px;
    }

    [data-testid="stFileUploaderDropzone"] * {
        color: #f8fafc !important;
    }

    /* Uploaded file name */
    [data-testid="stFileUploaderFileName"] {
        color: #ffffff !important;
        font-weight: 600;
    }

    /* Small helper text like file limit */
    .stCaption, .stCaption p {
        color: #cbd5e1 !important;
    }

    /* Markdown headings inside cards */
    h1, h2, h3, h4, h5, h6 {
        color: #ffffff !important;
    }

    /* Dataframe area */
    div[data-testid="stDataFrame"] {
        border-radius: 12px;
        overflow: hidden;
    }

    /* Success/info/error text */
    div[data-testid="stAlertContainer"] * {
        color: inherit !important;
    }
            
    /* Fix Browse files button */
    [data-testid="stBaseButton-secondary"] {
        background: linear-gradient(90deg, #00c9a7, #0072ff) !important;
        color: white !important;
        border: none !important;
        border-radius: 10px !important;
        font-weight: 600 !important;
    }

    [data-testid="stBaseButton-secondary"]:hover {
        opacity: 0.9 !important;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">⚡ AutoML Pipeline</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Upload any CSV — get a trained model, metrics, SHAP analysis and PDF report instantly</div>', unsafe_allow_html=True)

uploaded = st.file_uploader("📂 Upload your CSV file", type=["csv"])

if uploaded:
    df = pd.read_csv(uploaded)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("<h3 style='color: white;'>📊 Data Preview</h3>", unsafe_allow_html=True)
    st.dataframe(df.head(10), use_container_width=True)
    st.caption(f"{df.shape[0]} rows × {df.shape[1]} columns — last column used as target")
    st.markdown('</div>', unsafe_allow_html=True)

    if st.button("⚡ Run AutoML Pipeline"):
        with st.spinner("Training 4 models..."):
            try:
                results_df, best_model, best_name, task, X_train, X_test, features = train_models(df)

                st.success(f"✅ Task detected: **{task.capitalize()}** — Best model: **{best_name}**")

                # ── Metrics ───────────────────────────────────────────────────
                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.markdown("### 📈 Model Comparison")
                st.dataframe(results_df, use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)

                # ── SHAP ──────────────────────────────────────────────────────
                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.markdown("### 🔍 SHAP Feature Importance")
                fig = get_shap_plot(best_model, X_test, features)
                if fig:
                    st.pyplot(fig)
                else:
                    st.info("SHAP not available for this model type.")
                st.markdown('</div>', unsafe_allow_html=True)

                # ── Downloads ─────────────────────────────────────────────────
                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.markdown("### 📥 Downloads")

                col1, col2 = st.columns(2)

                with col1:
                    model_bytes = pickle.dumps(best_model)
                    st.download_button(
                        label="⬇️ Download Trained Model (.pkl)",
                        data=model_bytes,
                        file_name=f"{best_name.lower().replace(' ', '_')}_model.pkl",
                        mime="application/octet-stream"
                    )

                with col2:
                    pdf_bytes = generate_pdf(results_df, best_name, task)
                    st.download_button(
                        label="⬇️ Download PDF Report",
                        data=pdf_bytes,
                        file_name="automl_report.pdf",
                        mime="application/pdf"
                    )

                st.markdown('</div>', unsafe_allow_html=True)

            except Exception as e:
                st.error(f"❌ Something went wrong: {e}")