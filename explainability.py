# explainability.py
import shap
import lime
import lime.lime_tabular
import matplotlib.pyplot as plt

def explain_shap(model, X_sample):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)
    shap.summary_plot(shap_values, X_sample, plot_type="bar")

def explain_lime(model, X_train, X_sample):
    explainer = lime.lime_tabular.LimeTabularExplainer(X_train, mode="classification")
    for i in range(min(3, len(X_sample))):  # 只解释前3个样本，避免太多图形
        exp = explainer.explain_instance(X_sample[i], model.predict_proba, num_features=10)
        exp.show_in_notebook(show_table=True)
