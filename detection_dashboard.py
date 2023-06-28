from __future__ import print_function
import pandas as pd
import streamlit as st
import numpy as np
from dashboard_utils import load_data, explain_instance, get_severity
import pickle
import matplotlib.pyplot as plt


def main():
    # Load Data
    continuous_cols, kmeans, X_test, X_test_malicious, kmeans_test_y, y_test_preds_new, \
    y_test_benign, y_test_preds_combined, X_train_new2, X_test_new2, X_train_new, \
    X_test_new, X_train, y_train_benign, y_train, kmeans_test_labels, df_crosstab = load_data()

    st.title("Network Intrusion Detection Engine")

    with st.sidebar:
        "## Intrusion Alerts"
        st.dataframe(X_test_malicious[['sample_id', 'risk_score']], height=200)
        "There were " + str(len(X_test_malicious)) + " malicious samples detected."

        # Plot attacks by severity
        df_sev_counts = pd.DataFrame(X_test_malicious['Severity'].value_counts(),
                                     index=['Critical', 'High', 'Medium', 'Low'])
        print(df_sev_counts.head())
        x_pos = np.arange(len(df_sev_counts['Severity'].values))
        fig = plt.figure(figsize=(4, 3))
        plt.bar(x_pos, df_sev_counts['Severity'].values, color=['Red', '#F15E1C', '#F1891C', '#F1DB1C'])
        plt.xticks(x_pos, df_sev_counts.index)
        st.pyplot(fig)

        # Select sample to analyze
        severity_filter = st.selectbox('Filter by Severity (Optional)', ['None', 'Critical', 'High', 'Medium', 'Low'])
        if severity_filter != "None":
            X_test_malicious = X_test_malicious[X_test_malicious['Severity'] == severity_filter]
        selected_attack = st.selectbox('See Alert Analytics:', X_test_malicious['sample_id'])

    col1, col2 = st.columns(2)
    with col1:
        risk_score = round(X_test_malicious[X_test_malicious['sample_id'] == selected_attack]['risk_score'].values[0], 1)
        severity = get_severity(risk_score)
        color_map = {
            'Low': '#F1DB1C',
            'Medium': '#F1891C',
            'High': '#F15E1C',
            'Critical': 'Red'
        }
        st.markdown(
                '<p style="font-family:sans-serif; color:#707070; font-size: 20px;">Risk Score</p>',
                unsafe_allow_html=True)

        st.markdown('<span style="font-family:sans-serif; color:%s; font-size: 50px;">%.2f</span><span style="font-family:sans-serif; color:Black; font-size: 15px;">&nbsp;&nbsp;<i>%s</i></span>'%(color_map[severity], risk_score, severity), unsafe_allow_html=True)

    # Fit Models
    xgb_pipeline = pickle.load(open('models/xgb_pipeline.sav', 'rb'))
    xgb_pipeline2 = pickle.load(open('models/xgb_pipeline2.sav', 'rb'))


    i = int(selected_attack.strip('SID')) - 1
    explain_instance(i, kmeans, kmeans_test_y, y_test_preds_new, kmeans_test_labels, y_test_benign,
                     y_test_preds_combined, df_crosstab, continuous_cols, X_test, X_train, xgb_pipeline, xgb_pipeline2,
                     X_train_new, X_test_new, X_test_new2, col1, col2, optimal_threshold=0.0050058886)


if __name__ == '__main__':
    main()