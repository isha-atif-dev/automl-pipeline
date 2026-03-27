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

    header {
        visibility: hidden;
        height: 0px;
    }

    div[data-testid="stDecoration"] {
        display: none !important;
    }

    p, label, div, span {
        color: #f8fafc;
    }

    [data-testid="stFileUploaderDropzone"] {
        background: rgba(255,255,255,0.06);
        border: 1px dashed rgba(255,255,255,0.25);
        border-radius: 14px;
    }

    [data-testid="stFileUploaderDropzone"] * {
        color: #f8fafc !important;
    }

    [data-testid="stFileUploader"] small,
    [data-testid="stFileUploader"] span,
    [data-testid="stFileUploaderFile"] *,
    [data-testid="stFileUploaderFileName"] {
        color: #f8fafc !important;
        font-weight: 600;
    }

    .stCaption, .stCaption p {
        color: #cbd5e1 !important;
    }

    h1, h2, h3, h4, h5, h6 {
        color: #ffffff !important;
    }

    div[data-testid="stDataFrame"] {
        border-radius: 12px;
        overflow: hidden;
    }

    div[data-testid="stAlertContainer"] * {
        color: inherit !important;
    }

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

    [data-testid="stFileUploader"] svg,
    [data-testid="stDataFrame"] svg,
    [data-testid="stDownloadButton"] svg {
        fill: #f8fafc !important;
        color: #f8fafc !important;
    }

    /* Fullscreen button outer circle/container */
    button[title="View fullscreen"],
    [data-testid="StyledFullScreenButton"],
    [data-testid="StyledFullScreenButton"] button,
    div[data-testid="StyledFullScreenButton"] {
        background: rgba(255,255,255,0.10) !important;
        color: #cbd5e1 !important;
        border: 1px solid rgba(255,255,255,0.15) !important;
        border-radius: 999px !important;
        box-shadow: none !important;
    }

    button[title="View fullscreen"]:hover,
    [data-testid="StyledFullScreenButton"]:hover,
    [data-testid="StyledFullScreenButton"] button:hover {
        background: rgba(255,255,255,0.16) !important;
        color: #ffffff !important;
    }

    /* Fullscreen icon itself */
    button[title="View fullscreen"] svg,
    [data-testid="StyledFullScreenButton"] svg,
    [data-testid="StyledFullScreenButton"] button svg,
    [title="View fullscreen"] svg {
        fill: #cbd5e1 !important;
        color: #cbd5e1 !important;
        stroke: #cbd5e1 !important;
    }

    button[title="View fullscreen"]:hover svg,
    [data-testid="StyledFullScreenButton"]:hover svg,
    [data-testid="StyledFullScreenButton"] button:hover svg {
        fill: #ffffff !important;
        color: #ffffff !important;
        stroke: #ffffff !important;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">⚡ AutoML Pipeline</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">Upload any CSV — get a trained model, metrics, SHAP analysis and PDF report instantly</div>',
    unsafe_allow_html=True
)
st.info("💡 For best results on this free hosted version, upload CSV files under **5MB**. For larger files, run the app locally.")
uploaded = st.file_uploader("📂 Upload your CSV file", type=["csv"])


if uploaded:
    df = pd.read_csv(uploaded)
    file_size_mb = uploaded.size / (1024 * 1024)
    if file_size_mb > 50:
        st.warning(f"⚠️ File is {file_size_mb:.1f}MB. For best performance on this free tier, keep files under 50MB.")
    
    if len(df) > 50000:
        df = df.sample(n=50000, random_state=42)
        st.info(f"ℹ️ Large dataset detected. Sampled 50,000 rows for faster processing.")

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("<h3 style='color: white;'>📊 Data Preview</h3>", unsafe_allow_html=True)
    st.dataframe(df.head(10), width="stretch")
    st.caption(f"{df.shape[0]} rows × {df.shape[1]} columns — last column used as target")
    st.markdown('</div>', unsafe_allow_html=True)

    if st.button("⚡ Run AutoML Pipeline"):
        with st.spinner("Training 4 models..."):
            try:
                results_df, best_model, best_name, task, X_train, X_test, features = train_models(df)

                st.success(f"✅ Task detected: **{task.capitalize()}** — Best model: **{best_name}**")

                # Model comparison
                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.markdown("### 📈 Model Comparison")
                st.dataframe(results_df, width="stretch")
                st.markdown('</div>', unsafe_allow_html=True)

                # SHAP
                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.markdown("### 🔍 SHAP Feature Importance")
                fig = get_shap_plot(best_model, X_test, features)
                if fig is not None:
                    st.pyplot(fig)
                else:
                    st.info("SHAP could not be generated for this dataset/model combination.")
                st.markdown('</div>', unsafe_allow_html=True)

                # Downloads
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