import shap
import matplotlib.pyplot as plt
import numpy as np

def get_shap_plot(model, X_test, feature_names: list):
    """Generate SHAP feature importance bar chart."""
    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test)

        if isinstance(shap_values, list):
            shap_values = shap_values[1]

        mean_shap = np.abs(shap_values).mean(axis=0)

        fig, ax = plt.subplots(figsize=(8, 5))
        fig.patch.set_facecolor("#0d1b2a")
        ax.set_facecolor("#0d1b2a")

        sorted_idx = np.argsort(mean_shap)
        bars = ax.barh(
            [feature_names[i] for i in sorted_idx],
            mean_shap[sorted_idx],
            color="#00c9a7"
        )

        ax.set_xlabel("Mean |SHAP Value|", color="#a0aec0")
        ax.set_title("Feature Importance (SHAP)", color="#ffffff", fontsize=13)
        ax.tick_params(colors="#a0aec0")
        for spine in ax.spines.values():
            spine.set_edgecolor("#2d3748")

        plt.tight_layout()
        return fig

    except Exception:
        return None