import shap
import matplotlib.pyplot as plt
import numpy as np

def get_shap_plot(model, X_test, feature_names: list):
    """Generate SHAP feature importance bar chart."""
    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test)

        # Debug prints
        print("Model type:", type(model))
        print("SHAP values type:", type(shap_values))

        # Handle older SHAP multiclass output: list of arrays
        if isinstance(shap_values, list):
            print("SHAP list shapes:", [sv.shape for sv in shap_values])
            shap_matrix = np.mean([np.abs(sv) for sv in shap_values], axis=0)

        # Handle numpy array output
        elif isinstance(shap_values, np.ndarray):
            print("SHAP array shape:", shap_values.shape)

            if shap_values.ndim == 3:
                # Usually (samples, features, classes)
                shap_matrix = np.abs(shap_values).mean(axis=2)
            elif shap_values.ndim == 2:
                # Usually (samples, features)
                shap_matrix = np.abs(shap_values)
            else:
                return None

        else:
            return None

        mean_shap = shap_matrix.mean(axis=0)

        if len(mean_shap) != len(feature_names):
            print("SHAP shape mismatch:", mean_shap.shape, len(feature_names))
            return None

        # Show only top features so plot stays readable
        top_n = min(15, len(feature_names))
        sorted_idx = np.argsort(mean_shap)[-top_n:]

        fig, ax = plt.subplots(figsize=(10, 7))
        fig.patch.set_facecolor("#0d1b2a")
        ax.set_facecolor("#0d1b2a")

        ax.barh(
            [feature_names[i] for i in sorted_idx],
            mean_shap[sorted_idx],
            color="#00c9a7"
        )

        ax.set_xlabel("Mean |SHAP Value|", color="#a0aec0")
        ax.set_title("Top Feature Importance (SHAP)", color="#ffffff", fontsize=13)
        ax.tick_params(colors="#a0aec0")
        for spine in ax.spines.values():
            spine.set_edgecolor("#2d3748")

        plt.tight_layout()
        return fig

    except Exception as e:
        print(f"SHAP error: {e}")
        return None