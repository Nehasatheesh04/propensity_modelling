import pandas as pd
import joblib
import numpy as np
import shap
import matplotlib.pyplot as plt
import streamlit as st
from groq import Groq 
import os 
from dotenv import load_dotenv

load_dotenv()

# --- 1. CONFIGURATION & PATHS ---
MODEL_PATH = 'models/best_xgboost_classifier.joblib'
STATS_PATH = 'models/original_numerical_stats.joblib'
FEATURES_PATH = 'data/X_selected_processed.csv'
Y_PATH = 'data/y_resampled_processed.csv'

# Configure Groq (Free Tier Alternative to Claude)
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

st.set_page_config(page_title="MutualIQ Propensity Analyzer", layout="wide")

st.markdown("""
    <style>
    .score-card {
        background-color: #2563eb; color: white; padding: 20px;
        border-radius: 12px; text-align: center; margin-bottom: 15px;
    }
    .ai-card {
        background-color: #ffffff; border: 1px solid #e2e8f0;
        padding: 20px; border-radius: 12px; color: #1e293b;
    }
    .factor-label { font-weight: 500; font-size: 13px; margin-bottom: 2px; }
    </style>
""", unsafe_allow_html=True)

# --- 2. LOAD RESOURCES ---
@st.cache_resource
def load_resources():
    model = joblib.load(MODEL_PATH)
    stats = joblib.load(STATS_PATH) 
    X_df = pd.read_csv(FEATURES_PATH)
    y_df = pd.read_csv(Y_PATH)
    explainer = shap.TreeExplainer(model) 
    return model, stats, X_df, y_df, explainer

model, stats, X_selected_df, y_resampled_df, explainer = load_resources()

# --- 3. HELPER FUNCTIONS ---

def format_feature_name(name):
    """Removes technical suffixes for RM readability """
    clean_name = name.replace('_encoded', '').replace('_processed', '').replace('_', ' ')
    return clean_name.title()

def get_ai_summary(prob, top7):
    """Generates the 'Why this score' paragraph """
    lines = "\n".join([
        f"• {format_feature_name(name)}: {val} -> {'increases' if sv > 0 else 'decreases'} likelihood"
        for name, sv, val in top7
    ]) 
    
    prompt = (
        f"You are a friendly financial advisor explaining to a bank RM why a customer "
        f"has a {prob:.0%} chance of buying a mutual fund.\n\n"
        f"Key factors:\n{lines}\n\n"
        f"In 3-4 sentences explain why in plain English. Avoid technical terms like SHAP, "
        f"model, feature importance, or propensity score." 
    )
    
    chat = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}]
    )
    return chat.choices[0].message.content

def get_talking_points(prob, top7):
    """Generates 2 persuasive talking points """
    lines = "\n".join([f"• {format_feature_name(name)}: {val}" for name, sv, val in top7])
    prompt = (
        f"Based on a {prob:.0%} propensity score and these factors:\n{lines}\n"
        f"Provide 2 short talking points for a bank relationship manager to use on a call. "
        f"Focus on benefits like tax savings or wealth building. No jargon." 
    )
    
    chat = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}]
    )
    return chat.choices[0].message.content

# --- 4. STEP 1: CUSTOMER SELECTION ---
st.title("🎯 Mutual Fund Propensity Analyzer")
st.caption("AI-powered scoring with plain-English explanation")

preview_df = X_selected_df.head(10).copy()
for col in preview_df.columns:
    if col in stats:
        preview_df[col] = (preview_df[col] * stats[col]['std']) + stats[col]['mean']

selection = st.dataframe(preview_df, on_select="rerun", selection_mode="single-row", use_container_width=True)

if selection.selection.rows:
    row_idx = selection.selection.rows[0]
    row_data = preview_df.iloc[row_idx]
    
    input_real = {f: row_data[f] for f in X_selected_df.columns}
    scaled = {f: (input_real[f] - stats[f]['mean'])/stats[f]['std'] if f in stats else input_real[f] for f in X_selected_df.columns}
    final_df = pd.DataFrame([scaled])[X_selected_df.columns]
    
    prob = model.predict_proba(final_df)[:, 1][0]
    shap_v = explainer(final_df)
    
    pairs = sorted(zip(X_selected_df.columns, shap_v[0].values, input_real.values()), key=lambda x: abs(x[1]), reverse=True)
    top7 = pairs[:7]

    # Calculate Total Impact for Percentage Normalization
    total_abs_impact = sum(abs(sv) for name, sv, val in top7)

    st.divider()

    # --- 5. PROGRESSIVE UI LAYERS ---
    left_col, right_col = st.columns([1, 1], gap="large") 

    with left_col:
        # Layer 1: Propensity Score Card
        badge = "Strong Prospect" if prob > 0.7 else "Moderate" if prob > 0.4 else "Weak"
        st.markdown(f"""<div class="score-card">
            <small>Propensity Score</small>
            <h1>{prob:.0%}</h1>
            <span style="background:rgba(255,255,255,0.2); padding:2px 12px; border-radius:15px;">{badge}</span>
        </div>""", unsafe_allow_html=True)

        # Layer 2: AI Summary Card
        with st.container(border=True):
            st.subheader("✨ AI SUMMARY — WHY THIS SCORE")
            with st.spinner("Generating summary..."):
                summary = get_ai_summary(prob, top7)
                st.write(summary)
            
            c_btn1, c_btn2 = st.columns(2)
            if c_btn1.button("🔄 Regenerate explanation", use_container_width=True):
                st.rerun() 
            
            if c_btn2.button("💬 Get Talking Points", type="primary", use_container_width=True):
                with st.spinner("Consulting AI advisor..."):
                    points = get_talking_points(prob, top7)
                    st.info(points)

    with right_col:
        # Layer 3: Top Factors Bar Chart (Percentage and Reflected Bar)
        st.subheader('Top Factors Driving the Score')
        st.caption("Bars reflect the relative percentage contribution to the score")
        
        for name, sv, val in top7:
            # Calculate the relative percentage of impact
            relative_pct = (abs(sv) / total_abs_impact) if total_abs_impact > 0 else 0
            
            c1, c2, c3 = st.columns([3, 1, 1.2])
            
            # Progress bar reflects the relative impact percentage
            # We scale it so 0.5 impact (very high) shows a full bar for UX
            c1.progress(float(min(relative_pct * 2, 1.0)), text=format_feature_name(name))
            
            # Format value for display
            display_val = str(int(val) if val.is_integer() else f"{val:.1f}")
            c2.write(display_val)
            
            # Styling positive/negative markers with percentage
            impact_display = f"{relative_pct:.0%}"
            if sv > 0:
                c3.write(f":green[+{impact_display} ▲]")
            else:
                c3.write(f":red[-{impact_display} ▼]")

        st.divider()

        # Layer 4: Technical Waterfall (Collapsed)
        with st.expander('Technical SHAP waterfall (analysts only)'):
            fig, ax = plt.subplots(figsize=(8, 4))
            shap.plots.waterfall(shap_v[0], max_display=7, show=False)
            plt.tight_layout()
            st.pyplot(fig)

if st.button("Reset Demo"):
    st.rerun()