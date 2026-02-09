"""
🩺 DiabetesAI — Interactive Diabetes Prediction Laboratory
============================================================
A comprehensive ML-powered web app for predicting diabetes risk
using multiple machine learning models with full evaluation metrics.

Author : WarnAI
Framework: Streamlit
Dataset : Pima Indians Diabetes Database
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO

# Scikit-learn imports
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    confusion_matrix, accuracy_score, precision_score, recall_score,
    f1_score, roc_curve, auc, precision_recall_curve, log_loss,
    classification_report, roc_auc_score
)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier,
    AdaBoostClassifier, ExtraTreesClassifier
)
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
# Page Configuration
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="DiabetesAI — Prediction Lab",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────
# Custom CSS for polished UI
# ─────────────────────────────────────────────
st.markdown("""
<style>
    /* Import distinctive fonts */
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:ital,opsz,wght@0,9..40,300;0,9..40,500;0,9..40,700;1,9..40,400&family=JetBrains+Mono:wght@400;600&family=Playfair+Display:wght@700;900&display=swap');

    /* Root variables */
    :root {
        --primary: #0ea5e9;
        --primary-dark: #0284c7;
        --accent: #f43f5e;
        --success: #10b981;
        --warning: #f59e0b;
        --bg-card: rgba(15, 23, 42, 0.6);
        --border: rgba(148, 163, 184, 0.12);
    }

    /* Global overrides */
    .stApp {
        font-family: 'DM Sans', sans-serif;
    }

    /* Hero header */
    .hero-header {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #0f172a 100%);
        border: 1px solid var(--border);
        border-radius: 20px;
        padding: 2.5rem 3rem;
        margin-bottom: 2rem;
        position: relative;
        overflow: hidden;
    }
    .hero-header::before {
        content: '';
        position: absolute;
        top: -50%;
        right: -20%;
        width: 400px;
        height: 400px;
        background: radial-gradient(circle, rgba(14,165,233,0.15) 0%, transparent 70%);
        border-radius: 50%;
    }
    .hero-header h1 {
        font-family: 'Playfair Display', serif;
        font-size: 2.6rem;
        font-weight: 900;
        background: linear-gradient(135deg, #e2e8f0, #0ea5e9);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.3rem;
    }
    .hero-header p {
        color: #94a3b8;
        font-size: 1.05rem;
        margin: 0;
    }

    /* Metric cards */
    .metric-card {
        background: linear-gradient(145deg, rgba(15,23,42,0.8), rgba(30,41,59,0.6));
        border: 1px solid var(--border);
        border-radius: 16px;
        padding: 1.5rem;
        text-align: center;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .metric-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 12px 40px rgba(14,165,233,0.1);
    }
    .metric-card .metric-value {
        font-family: 'JetBrains Mono', monospace;
        font-size: 2rem;
        font-weight: 700;
        color: #e2e8f0;
    }
    .metric-card .metric-label {
        font-size: 0.82rem;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        margin-top: 0.4rem;
    }

    /* Result badges */
    .result-positive {
        background: linear-gradient(135deg, rgba(244,63,94,0.15), rgba(244,63,94,0.05));
        border: 1px solid rgba(244,63,94,0.3);
        border-radius: 16px;
        padding: 2rem;
        text-align: center;
    }
    .result-negative {
        background: linear-gradient(135deg, rgba(16,185,129,0.15), rgba(16,185,129,0.05));
        border: 1px solid rgba(16,185,129,0.3);
        border-radius: 16px;
        padding: 2rem;
        text-align: center;
    }

    /* Section headers */
    .section-header {
        font-family: 'DM Sans', sans-serif;
        font-size: 1.3rem;
        font-weight: 700;
        color: #e2e8f0;
        padding-bottom: 0.6rem;
        border-bottom: 2px solid var(--primary);
        margin: 2rem 0 1.2rem 0;
        display: inline-block;
    }

    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f172a 0%, #1e293b 100%);
    }
    section[data-testid="stSidebar"] .stMarkdown p {
        color: #cbd5e1;
    }

    /* Info boxes */
    .info-box {
        background: rgba(14,165,233,0.08);
        border-left: 4px solid var(--primary);
        border-radius: 0 12px 12px 0;
        padding: 1rem 1.4rem;
        margin: 1rem 0;
        color: #94a3b8;
        font-size: 0.92rem;
    }

    /* Table styling */
    .dataframe {
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 0.85rem !important;
    }

    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 10px 10px 0 0;
        padding: 10px 20px;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# Data Loading & Preprocessing
# ─────────────────────────────────────────────
@st.cache_data
def load_data():
    """Load and preprocess the Pima Indians Diabetes dataset."""
    df = pd.read_csv("diabetes.csv")
    return df

@st.cache_data
def preprocess_data(df):
    """Handle zero values and prepare features/target."""
    df_clean = df.copy()

    # Replace zeros with NaN for columns where 0 is biologically impossible
    zero_cols = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
    for col in zero_cols:
        df_clean[col] = df_clean[col].replace(0, np.nan)

    # Impute with median grouped by Outcome
    for col in zero_cols:
        df_clean[col] = df_clean.groupby('Outcome')[col].transform(
            lambda x: x.fillna(x.median())
        )

    return df_clean

@st.cache_data
def prepare_model_data(df_clean, test_size=0.2, random_state=42):
    """Split data and scale features."""
    X = df_clean.drop('Outcome', axis=1)
    y = df_clean['Outcome']
    feature_names = X.columns.tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, feature_names

# ─────────────────────────────────────────────
# Model Definitions
# ─────────────────────────────────────────────
MODEL_REGISTRY = {
    "Logistic Regression": {
        "model": LogisticRegression(max_iter=1000, random_state=42),
        "type": "Linear",
        "complexity": "⭐",
        "desc": "Baseline linear classifier using sigmoid function for binary classification."
    },
    "K-Nearest Neighbors": {
        "model": KNeighborsClassifier(n_neighbors=7),
        "type": "Instance-based",
        "complexity": "⭐⭐",
        "desc": "Classifies based on majority vote of K nearest training samples."
    },
    "Naive Bayes": {
        "model": GaussianNB(),
        "type": "Probabilistic",
        "complexity": "⭐",
        "desc": "Probabilistic classifier assuming feature independence (Bayes' theorem)."
    },
    "Decision Tree": {
        "model": DecisionTreeClassifier(max_depth=5, random_state=42),
        "type": "Tree-based",
        "complexity": "⭐⭐",
        "desc": "Splits data recursively by best feature thresholds to form a tree."
    },
    "Random Forest": {
        "model": RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42),
        "type": "Ensemble",
        "complexity": "⭐⭐⭐",
        "desc": "Ensemble of decision trees trained on bootstrap samples with random feature subsets."
    },
    "Gradient Boosting": {
        "model": GradientBoostingClassifier(n_estimators=200, max_depth=4, learning_rate=0.1, random_state=42),
        "type": "Ensemble",
        "complexity": "⭐⭐⭐⭐",
        "desc": "Sequentially builds weak learners, each correcting errors of the previous one."
    },
    "AdaBoost": {
        "model": AdaBoostClassifier(n_estimators=150, learning_rate=0.8, random_state=42, algorithm='SAMME'),
        "type": "Ensemble",
        "complexity": "⭐⭐⭐",
        "desc": "Adaptively boosts weak classifiers by reweighting misclassified samples."
    },
    "Support Vector Machine": {
        "model": SVC(kernel='rbf', probability=True, random_state=42),
        "type": "Kernel",
        "complexity": "⭐⭐⭐",
        "desc": "Finds optimal hyperplane boundary with maximum margin using kernel trick."
    },
    "Extra Trees": {
        "model": ExtraTreesClassifier(n_estimators=200, max_depth=8, random_state=42),
        "type": "Ensemble",
        "complexity": "⭐⭐⭐",
        "desc": "Like Random Forest but with random thresholds, reducing variance further."
    },
    "Neural Network (MLP)": {
        "model": MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500, random_state=42, early_stopping=True),
        "type": "Deep Learning",
        "complexity": "⭐⭐⭐⭐⭐",
        "desc": "Multi-layer perceptron with backpropagation — a simple artificial neural network."
    }
}

# ─────────────────────────────────────────────
# Model Training & Evaluation
# ─────────────────────────────────────────────
@st.cache_data
def train_and_evaluate(model_name, _X_train, _X_test, _y_train, _y_test):
    """Train a model and return all evaluation metrics."""
    model = MODEL_REGISTRY[model_name]["model"]
    model.fit(_X_train, _y_train)

    y_pred = model.predict(_X_test)
    y_proba = model.predict_proba(_X_test)[:, 1]

    # Core metrics
    acc = accuracy_score(_y_test, y_pred)
    prec = precision_score(_y_test, y_pred)
    rec = recall_score(_y_test, y_pred)
    f1 = f1_score(_y_test, y_pred)
    ll = log_loss(_y_test, y_proba)
    roc_auc_val = roc_auc_score(_y_test, y_proba)

    # Confusion matrix
    cm = confusion_matrix(_y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    specificity = tn / (tn + fp)

    # Curves
    fpr, tpr, _ = roc_curve(_y_test, y_proba)
    prec_curve, rec_curve, _ = precision_recall_curve(_y_test, y_proba)

    # Cross-validation
    cv_scores = cross_val_score(
        MODEL_REGISTRY[model_name]["model"], _X_train, _y_train, cv=5, scoring='accuracy'
    )

    return {
        "model": model,
        "y_pred": y_pred,
        "y_proba": y_proba,
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "log_loss": ll,
        "roc_auc": roc_auc_val,
        "specificity": specificity,
        "cm": cm,
        "fpr": fpr,
        "tpr": tpr,
        "prec_curve": prec_curve,
        "rec_curve": rec_curve,
        "cv_mean": cv_scores.mean(),
        "cv_std": cv_scores.std()
    }


# ─────────────────────────────────────────────
# Visualization Helpers
# ─────────────────────────────────────────────
PLOTLY_TEMPLATE = dict(
    layout=go.Layout(
        font=dict(family="DM Sans, sans-serif", color="#cbd5e1"),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(15,23,42,0.4)",
        xaxis=dict(gridcolor="rgba(148,163,184,0.08)", zerolinecolor="rgba(148,163,184,0.08)"),
        yaxis=dict(gridcolor="rgba(148,163,184,0.08)", zerolinecolor="rgba(148,163,184,0.08)"),
        margin=dict(l=40, r=20, t=50, b=40),
    )
)

def plot_confusion_matrix(cm, model_name):
    labels = ['Non-Diabetic (0)', 'Diabetic (1)']
    fig = go.Figure(data=go.Heatmap(
        z=cm[::-1],
        x=labels,
        y=labels[::-1],
        text=cm[::-1],
        texttemplate="%{text}",
        textfont=dict(size=20, family="JetBrains Mono"),
        colorscale=[[0, '#0f172a'], [0.5, '#0ea5e9'], [1, '#f43f5e']],
        showscale=False,
        hovertemplate="Actual: %{y}<br>Predicted: %{x}<br>Count: %{z}<extra></extra>"
    ))
    fig.update_layout(
        title=dict(text=f"Confusion Matrix — {model_name}", font=dict(size=16)),
        xaxis_title="Predicted Label",
        yaxis_title="Actual Label",
        height=420,
        **PLOTLY_TEMPLATE['layout'].to_plotly_json()
    )
    return fig

def plot_roc_curve(results_dict):
    fig = go.Figure()
    colors = px.colors.qualitative.Set2
    for i, (name, res) in enumerate(results_dict.items()):
        fig.add_trace(go.Scatter(
            x=res['fpr'], y=res['tpr'],
            name=f"{name} (AUC={res['roc_auc']:.3f})",
            line=dict(color=colors[i % len(colors)], width=2.5),
            hovertemplate="FPR: %{x:.3f}<br>TPR: %{y:.3f}<extra></extra>"
        ))
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1],
        name="Random (AUC=0.500)",
        line=dict(color="#475569", width=1.5, dash='dash'),
        showlegend=True
    ))
    fig.update_layout(
        title=dict(text="ROC Curve Comparison", font=dict(size=16)),
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        height=480,
        legend=dict(x=0.55, y=0.05, bgcolor="rgba(15,23,42,0.7)", bordercolor="rgba(148,163,184,0.2)", borderwidth=1),
        **PLOTLY_TEMPLATE['layout'].to_plotly_json()
    )
    return fig

def plot_precision_recall_curve(results_dict):
    fig = go.Figure()
    colors = px.colors.qualitative.Set2
    for i, (name, res) in enumerate(results_dict.items()):
        fig.add_trace(go.Scatter(
            x=res['rec_curve'], y=res['prec_curve'],
            name=name,
            line=dict(color=colors[i % len(colors)], width=2.5),
            hovertemplate="Recall: %{x:.3f}<br>Precision: %{y:.3f}<extra></extra>"
        ))
    fig.update_layout(
        title=dict(text="Precision–Recall Curve Comparison", font=dict(size=16)),
        xaxis_title="Recall",
        yaxis_title="Precision",
        height=480,
        legend=dict(x=0.02, y=0.05, bgcolor="rgba(15,23,42,0.7)", bordercolor="rgba(148,163,184,0.2)", borderwidth=1),
        **PLOTLY_TEMPLATE['layout'].to_plotly_json()
    )
    return fig

def plot_metrics_comparison(results_dict):
    metrics = ['accuracy', 'precision', 'recall', 'f1', 'specificity', 'roc_auc']
    metric_labels = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'Specificity', 'ROC AUC']
    models = list(results_dict.keys())

    fig = go.Figure()
    colors = px.colors.qualitative.Set2
    for i, model_name in enumerate(models):
        vals = [results_dict[model_name][m] for m in metrics]
        fig.add_trace(go.Scatterpolar(
            r=vals + [vals[0]],
            theta=metric_labels + [metric_labels[0]],
            name=model_name,
            line=dict(color=colors[i % len(colors)], width=2),
            fill='toself',
            fillcolor=f"rgba({','.join(str(int(c)) for c in px.colors.hex_to_rgb(colors[i % len(colors)]))},0.08)"
        ))
    fig.update_layout(
        title=dict(text="Model Performance Radar", font=dict(size=16, color="#cbd5e1")),
        polar=dict(
            bgcolor="rgba(15,23,42,0.4)",
            radialaxis=dict(visible=True, range=[0, 1], gridcolor="rgba(148,163,184,0.15)", tickfont=dict(color="#64748b")),
            angularaxis=dict(gridcolor="rgba(148,163,184,0.15)", tickfont=dict(color="#94a3b8"))
        ),
        height=520,
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="DM Sans", color="#cbd5e1"),
        legend=dict(bgcolor="rgba(15,23,42,0.7)", bordercolor="rgba(148,163,184,0.2)", borderwidth=1),
        margin=dict(l=60, r=60, t=60, b=40)
    )
    return fig


# ═══════════════════════════════════════════════
# MAIN APPLICATION
# ═══════════════════════════════════════════════
def main():
    # ─── Hero Header ───
    st.markdown("""
    <div class="hero-header">
        <h1>🩺 DiabetesAI</h1>
        <p>Interactive Machine Learning Laboratory for Diabetes Risk Prediction</p>
    </div>
    """, unsafe_allow_html=True)

    # ─── Load Data ───
    df_raw = load_data()
    df_clean = preprocess_data(df_raw)

    # ─── Sidebar ───
    with st.sidebar:
        st.markdown("### ⚙️ Configuration")
        st.markdown("---")

        page = st.radio(
            "📑 Navigation",
            ["🔬 Data Explorer", "🤖 Model Training", "🎯 Predict Diabetes", "📊 Model Comparison"],
            label_visibility="collapsed"
        )

        st.markdown("---")
        st.markdown("### 🧪 Training Settings")
        test_size = st.slider("Test split ratio", 0.10, 0.40, 0.20, 0.05)
        random_state = st.number_input("Random seed", 1, 999, 42)

        st.markdown("---")
        selected_models = st.multiselect(
            "🏗️ Select Models",
            list(MODEL_REGISTRY.keys()),
            default=["Logistic Regression", "Random Forest", "Gradient Boosting"]
        )

        st.markdown("---")
        st.markdown(
            '<div class="info-box">💡 Built with Streamlit • scikit-learn • Plotly<br>'
            'Dataset: Pima Indians Diabetes Database (NIDDK)</div>',
            unsafe_allow_html=True
        )

    # ─── Prepare data ───
    X_train, X_test, y_train, y_test, scaler, feature_names = prepare_model_data(
        df_clean, test_size, random_state
    )

    # ═══════════════════════════════════════
    # PAGE 1: DATA EXPLORER
    # ═══════════════════════════════════════
    if page == "🔬 Data Explorer":
        st.markdown('<div class="section-header">📋 Dataset Overview</div>', unsafe_allow_html=True)

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{df_raw.shape[0]}</div>
                <div class="metric-label">Total Samples</div>
            </div>""", unsafe_allow_html=True)
        with c2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{df_raw.shape[1] - 1}</div>
                <div class="metric-label">Features</div>
            </div>""", unsafe_allow_html=True)
        with c3:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{(df_raw['Outcome']==1).sum()}</div>
                <div class="metric-label">Diabetic Cases</div>
            </div>""", unsafe_allow_html=True)
        with c4:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{(df_raw['Outcome']==0).sum()}</div>
                <div class="metric-label">Non-Diabetic</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("")

        tab1, tab2, tab3 = st.tabs(["📊 Raw Data", "📈 Distributions", "🔗 Correlations"])

        with tab1:
            st.dataframe(df_raw, use_container_width=True, height=400)
            st.markdown(f"**Statistical Summary:**")
            st.dataframe(df_raw.describe().round(2), use_container_width=True)

        with tab2:
            feature = st.selectbox("Select Feature", feature_names)
            fig = px.histogram(
                df_raw, x=feature, color='Outcome',
                color_discrete_map={0: '#0ea5e9', 1: '#f43f5e'},
                barmode='overlay', opacity=0.75,
                labels={'Outcome': 'Diabetes'},
                title=f"Distribution of {feature} by Outcome"
            )
            fig.update_layout(**PLOTLY_TEMPLATE['layout'].to_plotly_json(), height=440)
            st.plotly_chart(fig, use_container_width=True)

            # Box plots
            fig2 = px.box(
                df_raw, x='Outcome', y=feature,
                color='Outcome',
                color_discrete_map={0: '#0ea5e9', 1: '#f43f5e'},
                title=f"Box Plot of {feature} by Outcome"
            )
            fig2.update_layout(**PLOTLY_TEMPLATE['layout'].to_plotly_json(), height=400)
            st.plotly_chart(fig2, use_container_width=True)

        with tab3:
            corr = df_clean.corr()
            fig = go.Figure(data=go.Heatmap(
                z=corr.values,
                x=corr.columns,
                y=corr.columns,
                text=corr.values.round(2),
                texttemplate="%{text}",
                textfont=dict(size=11),
                colorscale=[[0, '#0ea5e9'], [0.5, '#0f172a'], [1, '#f43f5e']],
                zmid=0,
                hovertemplate="%{x} vs %{y}: %{z:.3f}<extra></extra>"
            ))
            fig.update_layout(
                title=dict(text="Feature Correlation Heatmap", font=dict(size=16)),
                height=550,
                **PLOTLY_TEMPLATE['layout'].to_plotly_json()
            )
            st.plotly_chart(fig, use_container_width=True)

            st.markdown("""
            <div class="info-box">
            <strong>Key Observations:</strong><br>
            • Glucose has the strongest correlation with Outcome (diabetes).<br>
            • BMI, Age, and Pregnancies also show notable positive correlations.<br>
            • Insulin and SkinThickness are highly correlated with each other.
            </div>
            """, unsafe_allow_html=True)


    # ═══════════════════════════════════════
    # PAGE 2: MODEL TRAINING
    # ═══════════════════════════════════════
    elif page == "🤖 Model Training":
        st.markdown('<div class="section-header">🏗️ Train & Evaluate Models</div>', unsafe_allow_html=True)

        if not selected_models:
            st.warning("⚠️ Please select at least one model from the sidebar.")
            return

        # Train all selected models
        all_results = {}
        progress = st.progress(0, text="Training models...")
        for i, model_name in enumerate(selected_models):
            progress.progress((i + 1) / len(selected_models), text=f"Training {model_name}...")
            all_results[model_name] = train_and_evaluate(
                model_name, X_train, X_test, y_train, y_test
            )
        progress.empty()

        st.success(f"✅ Successfully trained {len(selected_models)} model(s)!")

        # ── Per-model results in tabs ──
        tabs = st.tabs([f"📋 {m}" for m in selected_models])
        for tab, model_name in zip(tabs, selected_models):
            with tab:
                res = all_results[model_name]
                info = MODEL_REGISTRY[model_name]

                # Model info
                st.markdown(f"""
                <div class="info-box">
                    <strong>{model_name}</strong> &nbsp;{info['complexity']}<br>
                    Type: {info['type']} &nbsp;|&nbsp; {info['desc']}
                </div>
                """, unsafe_allow_html=True)

                # Metric cards row
                m1, m2, m3, m4, m5, m6 = st.columns(6)
                for col, label, val, fmt in [
                    (m1, "Accuracy", res['accuracy'], ".1%"),
                    (m2, "Precision", res['precision'], ".1%"),
                    (m3, "Recall", res['recall'], ".1%"),
                    (m4, "F1-Score", res['f1'], ".1%"),
                    (m5, "Specificity", res['specificity'], ".1%"),
                    (m6, "ROC AUC", res['roc_auc'], ".3f"),
                ]:
                    with col:
                        display = f"{val:{fmt}}" if '%' in fmt else f"{val:{fmt}}"
                        st.markdown(f"""
                        <div class="metric-card">
                            <div class="metric-value">{display}</div>
                            <div class="metric-label">{label}</div>
                        </div>""", unsafe_allow_html=True)

                st.markdown("")
                extra1, extra2 = st.columns(2)
                with extra1:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-value">{res['log_loss']:.4f}</div>
                        <div class="metric-label">Log Loss</div>
                    </div>""", unsafe_allow_html=True)
                with extra2:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-value">{res['cv_mean']:.1%} ± {res['cv_std']:.1%}</div>
                        <div class="metric-label">Cross-Val Accuracy (5-Fold)</div>
                    </div>""", unsafe_allow_html=True)

                st.markdown("")

                # Charts
                ch1, ch2 = st.columns(2)
                with ch1:
                    st.plotly_chart(plot_confusion_matrix(res['cm'], model_name), use_container_width=True)
                with ch2:
                    # Single model ROC
                    fig_roc = go.Figure()
                    fig_roc.add_trace(go.Scatter(
                        x=res['fpr'], y=res['tpr'],
                        name=f"AUC = {res['roc_auc']:.3f}",
                        line=dict(color='#0ea5e9', width=3),
                        fill='tozeroy', fillcolor='rgba(14,165,233,0.1)'
                    ))
                    fig_roc.add_trace(go.Scatter(
                        x=[0, 1], y=[0, 1],
                        line=dict(color='#475569', dash='dash', width=1.5),
                        showlegend=False
                    ))
                    fig_roc.update_layout(
                        title=dict(text=f"ROC Curve — {model_name}", font=dict(size=16)),
                        xaxis_title="False Positive Rate",
                        yaxis_title="True Positive Rate",
                        height=420,
                        legend=dict(x=0.6, y=0.05),
                        **PLOTLY_TEMPLATE['layout'].to_plotly_json()
                    )
                    st.plotly_chart(fig_roc, use_container_width=True)

                # Precision-Recall curve
                fig_pr = go.Figure()
                fig_pr.add_trace(go.Scatter(
                    x=res['rec_curve'], y=res['prec_curve'],
                    line=dict(color='#f43f5e', width=3),
                    fill='tozeroy', fillcolor='rgba(244,63,94,0.08)',
                    name=model_name
                ))
                fig_pr.update_layout(
                    title=dict(text=f"Precision–Recall Curve — {model_name}", font=dict(size=16)),
                    xaxis_title="Recall", yaxis_title="Precision",
                    height=400,
                    **PLOTLY_TEMPLATE['layout'].to_plotly_json()
                )
                st.plotly_chart(fig_pr, use_container_width=True)

        # Store for comparison page
        st.session_state['all_results'] = all_results


    # ═══════════════════════════════════════
    # PAGE 3: PREDICT DIABETES
    # ═══════════════════════════════════════
    elif page == "🎯 Predict Diabetes":
        st.markdown('<div class="section-header">🎯 Diabetes Risk Prediction</div>', unsafe_allow_html=True)

        pred_tab1, pred_tab2 = st.tabs(["✍️ Manual Input", "📁 Upload CSV"])

        # ── Manual Input ──
        with pred_tab1:
            st.markdown("""
            <div class="info-box">
            Enter patient data below. All values should be clinically plausible.
            </div>
            """, unsafe_allow_html=True)

            col1, col2, col3, col4 = st.columns(4)
            with col1:
                pregnancies = st.number_input("🤰 Pregnancies", 0, 20, 1, help="Number of pregnancies")
                insulin = st.number_input("💉 Insulin (μU/mL)", 0, 900, 80, help="2-hour serum insulin")
            with col2:
                glucose = st.number_input("🩸 Glucose (mg/dL)", 0, 250, 120, help="Plasma glucose (2h OGTT)")
                bmi = st.number_input("⚖️ BMI (kg/m²)", 0.0, 70.0, 25.0, 0.1, help="Body mass index")
            with col3:
                bp = st.number_input("💓 Blood Pressure (mm Hg)", 0, 140, 72, help="Diastolic blood pressure")
                dpf = st.number_input("🧬 Diabetes Pedigree", 0.0, 2.5, 0.47, 0.01, help="Diabetes pedigree function")
            with col4:
                skin = st.number_input("📏 Skin Thickness (mm)", 0, 100, 29, help="Triceps skin fold thickness")
                age = st.number_input("🎂 Age (years)", 21, 100, 33, help="Age in years")

            pred_model = st.selectbox("🤖 Select Prediction Model", list(MODEL_REGISTRY.keys()))

            if st.button("🔍 Predict Now", type="primary", use_container_width=True):
                # Train model
                res = train_and_evaluate(pred_model, X_train, X_test, y_train, y_test)
                model = res['model']

                # Prepare input
                input_data = np.array([[pregnancies, glucose, bp, skin, insulin, bmi, dpf, age]])
                input_scaled = scaler.transform(input_data)
                prediction = model.predict(input_scaled)[0]
                proba = model.predict_proba(input_scaled)[0]

                st.markdown("---")
                st.markdown("### 📋 Prediction Result")

                r1, r2 = st.columns([2, 1])
                with r1:
                    if prediction == 1:
                        st.markdown(f"""
                        <div class="result-positive">
                            <h2 style="color:#f43f5e; margin:0;">⚠️ HIGH RISK — Diabetic</h2>
                            <p style="color:#fda4af; font-size:1.1rem; margin-top:0.5rem;">
                                Confidence: <strong>{proba[1]*100:.1f}%</strong>
                            </p>
                            <p style="color:#94a3b8; font-size:0.9rem;">
                                The model predicts this patient is likely diabetic. Please consult a physician for confirmation.
                            </p>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div class="result-negative">
                            <h2 style="color:#10b981; margin:0;">✅ LOW RISK — Non-Diabetic</h2>
                            <p style="color:#6ee7b7; font-size:1.1rem; margin-top:0.5rem;">
                                Confidence: <strong>{proba[0]*100:.1f}%</strong>
                            </p>
                            <p style="color:#94a3b8; font-size:0.9rem;">
                                The model predicts this patient is unlikely to be diabetic. Regular check-ups recommended.
                            </p>
                        </div>
                        """, unsafe_allow_html=True)

                with r2:
                    fig_gauge = go.Figure(go.Indicator(
                        mode="gauge+number",
                        value=proba[1] * 100,
                        number=dict(suffix="%", font=dict(size=40, color="#e2e8f0")),
                        gauge=dict(
                            axis=dict(range=[0, 100], tickfont=dict(color="#64748b")),
                            bar=dict(color="#f43f5e" if prediction == 1 else "#10b981"),
                            bgcolor="rgba(15,23,42,0.4)",
                            bordercolor="rgba(148,163,184,0.2)",
                            steps=[
                                dict(range=[0, 30], color="rgba(16,185,129,0.15)"),
                                dict(range=[30, 60], color="rgba(245,158,11,0.15)"),
                                dict(range=[60, 100], color="rgba(244,63,94,0.15)")
                            ],
                            threshold=dict(line=dict(color="#e2e8f0", width=2), thickness=0.8, value=50)
                        ),
                        title=dict(text="Risk Score", font=dict(size=16, color="#94a3b8"))
                    ))
                    fig_gauge.update_layout(
                        height=280,
                        paper_bgcolor="rgba(0,0,0,0)",
                        margin=dict(l=20, r=20, t=50, b=20)
                    )
                    st.plotly_chart(fig_gauge, use_container_width=True)

                # Feature importance (for tree-based models)
                if hasattr(model, 'feature_importances_'):
                    fi = pd.DataFrame({
                        'Feature': feature_names,
                        'Importance': model.feature_importances_
                    }).sort_values('Importance', ascending=True)

                    fig_fi = px.bar(
                        fi, x='Importance', y='Feature', orientation='h',
                        color='Importance',
                        color_continuous_scale=[[0, '#0f172a'], [0.5, '#0ea5e9'], [1, '#f43f5e']],
                        title="Feature Importance"
                    )
                    fig_fi.update_layout(
                        height=380, showlegend=False,
                        coloraxis_showscale=False,
                        **PLOTLY_TEMPLATE['layout'].to_plotly_json()
                    )
                    st.plotly_chart(fig_fi, use_container_width=True)

        # ── CSV Upload ──
        with pred_tab2:
            st.markdown("""
            <div class="info-box">
            📁 Upload a CSV file with the same columns as the training data (without <code>Outcome</code>).<br>
            Required columns: <code>Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age</code>
            </div>
            """, unsafe_allow_html=True)

            uploaded_file = st.file_uploader("Upload CSV for batch prediction", type=['csv'])

            if uploaded_file is not None:
                try:
                    df_upload = pd.read_csv(uploaded_file)
                    st.markdown("**Preview of uploaded data:**")
                    st.dataframe(df_upload.head(10), use_container_width=True)

                    required_cols = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
                                     'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']

                    # Check for Outcome column (for evaluation)
                    has_outcome = 'Outcome' in df_upload.columns

                    missing = [c for c in required_cols if c not in df_upload.columns]
                    if missing:
                        st.error(f"❌ Missing columns: {', '.join(missing)}")
                    else:
                        batch_model_name = st.selectbox("Select Model for Batch Prediction", list(MODEL_REGISTRY.keys()), key='batch_model')

                        if st.button("🚀 Run Batch Prediction", type="primary", use_container_width=True):
                            res = train_and_evaluate(batch_model_name, X_train, X_test, y_train, y_test)
                            model = res['model']

                            X_new = df_upload[required_cols].values
                            X_new_scaled = scaler.transform(X_new)

                            preds = model.predict(X_new_scaled)
                            probas = model.predict_proba(X_new_scaled)[:, 1]

                            df_result = df_upload.copy()
                            df_result['Prediction'] = preds
                            df_result['Prediction_Label'] = df_result['Prediction'].map({0: 'Non-Diabetic', 1: 'Diabetic'})
                            df_result['Risk_Probability'] = (probas * 100).round(2)

                            st.markdown("### 📊 Batch Results")

                            bc1, bc2, bc3 = st.columns(3)
                            with bc1:
                                st.markdown(f"""
                                <div class="metric-card">
                                    <div class="metric-value">{len(df_result)}</div>
                                    <div class="metric-label">Total Patients</div>
                                </div>""", unsafe_allow_html=True)
                            with bc2:
                                st.markdown(f"""
                                <div class="metric-card">
                                    <div class="metric-value" style="color:#f43f5e;">{(preds==1).sum()}</div>
                                    <div class="metric-label">Predicted Diabetic</div>
                                </div>""", unsafe_allow_html=True)
                            with bc3:
                                st.markdown(f"""
                                <div class="metric-card">
                                    <div class="metric-value" style="color:#10b981;">{(preds==0).sum()}</div>
                                    <div class="metric-label">Predicted Non-Diabetic</div>
                                </div>""", unsafe_allow_html=True)

                            st.markdown("")
                            st.dataframe(df_result, use_container_width=True, height=400)

                            # If Outcome exists, show evaluation metrics
                            if has_outcome:
                                st.markdown("### 📏 Evaluation on Uploaded Data")
                                y_true = df_upload['Outcome']
                                acc = accuracy_score(y_true, preds)
                                prec = precision_score(y_true, preds, zero_division=0)
                                rec = recall_score(y_true, preds, zero_division=0)
                                f1_val = f1_score(y_true, preds, zero_division=0)
                                st.markdown(f"**Accuracy:** `{acc:.2%}` | **Precision:** `{prec:.2%}` | **Recall:** `{rec:.2%}` | **F1:** `{f1_val:.2%}`")

                            # Download button
                            csv_buf = df_result.to_csv(index=False)
                            st.download_button(
                                "⬇️ Download Results as CSV",
                                csv_buf,
                                "diabetes_predictions.csv",
                                "text/csv",
                                use_container_width=True
                            )

                except Exception as e:
                    st.error(f"❌ Error processing file: {str(e)}")


    # ═══════════════════════════════════════
    # PAGE 4: MODEL COMPARISON
    # ═══════════════════════════════════════
    elif page == "📊 Model Comparison":
        st.markdown('<div class="section-header">📊 Head-to-Head Model Comparison</div>', unsafe_allow_html=True)

        if not selected_models:
            st.warning("⚠️ Please select at least 2 models from the sidebar for comparison.")
            return

        # Train all selected
        all_results = {}
        progress = st.progress(0, text="Training models...")
        for i, model_name in enumerate(selected_models):
            progress.progress((i + 1) / len(selected_models), text=f"Training {model_name}...")
            all_results[model_name] = train_and_evaluate(model_name, X_train, X_test, y_train, y_test)
        progress.empty()

        # ── Summary Table ──
        summary_data = []
        for name, res in all_results.items():
            summary_data.append({
                "Model": name,
                "Accuracy": f"{res['accuracy']:.2%}",
                "Precision": f"{res['precision']:.2%}",
                "Recall": f"{res['recall']:.2%}",
                "F1-Score": f"{res['f1']:.2%}",
                "Specificity": f"{res['specificity']:.2%}",
                "ROC AUC": f"{res['roc_auc']:.4f}",
                "Log Loss": f"{res['log_loss']:.4f}",
                "CV Accuracy": f"{res['cv_mean']:.2%} ± {res['cv_std']:.2%}",
            })
        df_summary = pd.DataFrame(summary_data)
        st.dataframe(df_summary, use_container_width=True, hide_index=True)

        # ── Bar chart comparison ──
        metrics_for_bar = ['accuracy', 'precision', 'recall', 'f1', 'specificity', 'roc_auc']
        bar_data = []
        for name, res in all_results.items():
            for m in metrics_for_bar:
                bar_data.append({"Model": name, "Metric": m.replace('_', ' ').title(), "Value": res[m]})
        df_bar = pd.DataFrame(bar_data)

        fig_bar = px.bar(
            df_bar, x="Metric", y="Value", color="Model",
            barmode="group",
            color_discrete_sequence=px.colors.qualitative.Set2,
            title="Metric Comparison Across Models"
        )
        fig_bar.update_layout(
            height=450,
            yaxis=dict(range=[0, 1]),
            **PLOTLY_TEMPLATE['layout'].to_plotly_json()
        )
        st.plotly_chart(fig_bar, use_container_width=True)

        # ── Curves comparison ──
        comp1, comp2 = st.columns(2)
        with comp1:
            st.plotly_chart(plot_roc_curve(all_results), use_container_width=True)
        with comp2:
            st.plotly_chart(plot_precision_recall_curve(all_results), use_container_width=True)

        # ── Radar chart ──
        if len(selected_models) >= 2:
            st.plotly_chart(plot_metrics_comparison(all_results), use_container_width=True)

        # ── Best model highlight ──
        best_model = max(all_results, key=lambda x: all_results[x]['f1'])
        best_res = all_results[best_model]
        st.markdown(f"""
        <div class="info-box">
            🏆 <strong>Best Model (by F1-Score):</strong> {best_model}<br>
            Accuracy: {best_res['accuracy']:.2%} | F1: {best_res['f1']:.2%} | AUC: {best_res['roc_auc']:.4f} | Log Loss: {best_res['log_loss']:.4f}
        </div>
        """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
