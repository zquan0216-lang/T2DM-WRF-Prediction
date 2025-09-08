import streamlit as st
import joblib
import shap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 载入 XGBoost 模型
model = joblib.load('xgb_model.pkl')

# 特征定义
feature_ranges = {
    "eGFR": {"min": 0, "max": 900, "default": 73.5},
    "BMI": {"min": 10, "max": 60, "default": 34.39},
    "CMI": {"min": 0.0, "max": 70.0, "default": 2.13},
    "UAlb": {"min": 0, "max": 1000, "default": 0.66},
    "UCr": {"min": 0, "max": 700, "default": 152.2},
    "Age": {"min": 18, "max": 80, "default": 55.9},
    "potassium": {"min": 0, "max": 10, "default": 3.92},
}

# 标题
st.title("WRF Risk Prediction (XGBoost + SHAP)")

# 输入界面
feature_values = []
for feature, v in feature_ranges.items():
    val = st.number_input(
        f"{feature} ({v['min']} - {v['max']})",
        min_value=float(v["min"]),
        max_value=float(v["max"]),
        value=float(v["default"]),
    )
    feature_values.append(val)

input_df = pd.DataFrame([feature_values], columns=feature_ranges.keys())

# 预测 + 力图
if st.button("Predict and Show SHAP Force Plot"):
    # 预测类别
    pred_class = model.predict(input_df)[0]
    pred_proba = model.predict_proba(input_df)[0][pred_class]

    # 显示预测概率
    st.markdown(f"### Predicted risk of WRF: **{pred_proba*100:.2f}%**")

    # SHAP 力图
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(input_df)

    # 生成 HTML 力图
    force_html = shap.force_plot(
        explainer.expected_value,
        shap_values[0],
        input_df.iloc[0],
        feature_names=input_df.columns
    )

    # 显示力图
    st.components.v1.html(shap.save_html("force_plot.html", force_html), height=300)
