import streamlit as st
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt

# 1. 加载模型
model = joblib.load('XGBoost.pkl')

# 2. 定义特征
selected_features = ["cmi", "gfr", "ucreat", "ualb", "fpg", "age"]

# 3. 页面标题
st.title("T2DM worsening renal function Risk Predictor")

# 4. 用户输入
cmi = st.number_input("cmi:", min_value=0.0, max_value=100.0, value=1.0)
gfr = st.number_input("gfr:", min_value=0.0, max_value=1000.0, value=90.0)
ucreat = st.number_input("ucreat:", min_value=0.0, max_value=1000.0, value=100.0)
ualb = st.number_input("ualb:", min_value=0.0, max_value=1000.0, value=30.0)
fpg = st.number_input("fpg:", min_value=0.0, max_value=1000.0, value=5.5)
age = st.number_input("age:", min_value=1, max_value=120, value=50)

# 5. 模型预测
input_values = [cmi, gfr, ucreat, ualb, fpg, age]
X_input = np.array([input_values])

if st.button("Predict"):
    # 预测
    pred_class = model.predict(X_input)[0]
    pred_proba = model.predict_proba(X_input)[0]

    st.write(f"**Predicted Class:** {pred_class}")
    st.write(f"**Prediction Probabilities:** {pred_proba}")

    prob = pred_proba[pred_class] * 100
    if pred_class == 1:
        st.warning(f"⚠️ High risk ({prob:.1f}%). This result indicates an increased risk. Please consider **closer monitoring, additional diagnostics, or early interventions** based on your clinical judgement.")
    else:
        st.success(f"✅ Low risk ({prob:.1f}%). Continue with **standard follow-up** and clinical monitoring as per current guidelines.")

    # 6. SHAP 可解释性
    df = pd.DataFrame([input_values], columns=selected_features)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(df)

    plt.figure(figsize=(10, 4), dpi=300)
    shap.force_plot(
        explainer.expected_value, 
        shap_values[0], 
        df.iloc[0], 
        feature_names=selected_features, 
        matplotlib=True, 
        show=False
    )
    plt.tight_layout()
    plt.savefig("force_plot.png", bbox_inches='tight', dpi=300)
    st.image("force_plot.png", caption="SHAP Force Plot")
