---
title: AutoML Pipeline
emoji: ⚡
colorFrom: blue
colorTo: indigo
sdk: docker
pinned: false
---

# ⚡ AutoML Pipeline Builder

An automated ML pipeline that takes any CSV and returns a trained model, metrics, SHAP analysis and PDF report — no coding required.

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![Streamlit](https://img.shields.io/badge/Built%20with-Streamlit-ff4b4b)
![SHAP](https://img.shields.io/badge/Explainability-SHAP-00c9a7)
![License](https://img.shields.io/badge/License-MIT-green)

## ✨ Features

- 🤖 **Auto Task Detection** — Automatically detects classification or regression
- 📊 **4 Model Comparison** — Trains Logistic/Linear Regression, Random Forest, XGBoost and LightGBM
- 🔍 **SHAP Explainability** — Feature importance chart for the best model
- 📥 **Downloadable Model** — Download the trained model as a `.pkl` file
- 📄 **PDF Report** — Full model comparison report as a downloadable PDF

## 🚀 Live Demo

👉 [Try it on HuggingFace Spaces](https://huggingface.co/spaces/ishaAtif/automl-pipeline)

## 🛠️ Tech Stack

| Tool | Purpose |
|---|---|
| Python | Core language |
| Streamlit | Web interface |
| Scikit-learn | ML models |
| XGBoost | Gradient boosting |
| LightGBM | Fast gradient boosting |
| SHAP | Model explainability |
| ReportLab | PDF generation |

## ⚙️ Run Locally
```bash
# Clone the repo
git clone https://github.com/isha-atif-dev/automl-pipeline.git
cd automl-pipeline

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

## 📁 Project Structure
```
automl-pipeline/
├── app.py            # Streamlit frontend
├── pipeline.py       # ML training logic
├── explainer.py      # SHAP explainability
├── reporter.py       # PDF report generation
├── requirements.txt  # Dependencies
├── Dockerfile        # HuggingFace deployment
└── .gitignore        # Ignores temp files
```

## 📊 Supported Datasets

Any CSV file where:
- Each row is one sample
- Each column is a feature
- The **last column** is the target variable

Works best with datasets between 100 and 100,000 rows.

## 🔍 How SHAP Works

SHAP (SHapley Additive exPlanations) explains **why** the model made each prediction by measuring how much each feature contributed to the output. This makes the model transparent and trustworthy — not a black box.

## 🔒 Note

No API keys required. All processing happens locally using open source libraries.