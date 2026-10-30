"""Streamlit app: Breast Cancer Diagnosis decision-support demo.

Run locally:  pip install -r requirements.txt && streamlit run app.py
The model is re-trained on start-up (takes about a second) with the same settings
chosen in the notebook, so the app never depends on a pickled model version.
"""
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, recall_score

THRESHOLD = 0.225  # recall-oriented threshold selected on training data in the notebook
SEED = 42

st.set_page_config(page_title="Breast Cancer Diagnosis", page_icon="🩺", layout="wide")


@st.cache_resource
def train():
    data = load_breast_cancer(as_frame=True)
    X, y = data.data, (data.target == 0).astype(int)  # 1 = malignant
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)
    model = Pipeline([("scale", StandardScaler()),
                      ("clf", LogisticRegression(C=1, penalty="l1", solver="liblinear", max_iter=5000))])
    model.fit(X_tr, y_tr)
    p = model.predict_proba(X_te)[:, 1]
    metrics = {"auc": roc_auc_score(y_te, p), "recall": recall_score(y_te, p >= THRESHOLD)}
    return model, X, y, X_te, y_te, metrics


model, X, y, X_te, y_te, metrics = train()
coefs = pd.Series(model.named_steps["clf"].coef_[0], index=X.columns)
key_features = coefs.abs().sort_values(ascending=False).head(8).index.tolist()

st.title("🩺 Breast Cancer Diagnosis — Decision-Support Demo")
st.caption("L1-regularised Logistic Regression on the Wisconsin Diagnostic Breast Cancer dataset · "
           "educational demo, not a medical device")

c1, c2, c3 = st.columns(3)
c1.metric("Test ROC-AUC", f"{metrics['auc']:.3f}")
c2.metric("Test recall (malignant)", f"{metrics['recall']:.1%}")
c3.metric("Decision threshold", f"{THRESHOLD:.2f}")

st.sidebar.header("Patient measurements")
mode = st.sidebar.radio("Start from", ["A test-set patient", "Median values"])
if mode == "A test-set patient":
    idx = st.sidebar.selectbox("Test patient", X_te.index.tolist(),
                               format_func=lambda i: f"#{i} (actual: {'malignant' if y.loc[i] else 'benign'})")
    base = X.loc[idx].copy()
else:
    base = X.median().copy()

st.sidebar.markdown("Adjust the **8 most influential** features:")
row = base.copy()
for f in key_features:
    lo, hi = float(X[f].min()), float(X[f].max())
    row[f] = st.sidebar.slider(f, lo, hi, float(base[f]), (hi - lo) / 200)

prob = float(model.predict_proba(row.to_frame().T)[:, 1][0])
malignant = prob >= THRESHOLD

left, right = st.columns([1, 1])
with left:
    st.subheader("Prediction")
    if malignant:
        st.error(f"**Likely MALIGNANT** — probability {prob:.1%}  (≥ {THRESHOLD:.0%} threshold)")
    else:
        st.success(f"**Likely BENIGN** — probability of malignancy {prob:.1%}  (< {THRESHOLD:.0%} threshold)")
    st.progress(min(prob, 1.0))
    st.caption("The threshold is deliberately low so that very few malignant cases are missed; "
               "borderline results should go for further clinical review.")
with right:
    st.subheader("Why? Feature contributions")
    z = model.named_steps["scale"].transform(row.to_frame().T)[0]
    contrib = pd.Series(z * coefs.values, index=X.columns)
    contrib = contrib[coefs != 0].sort_values(key=np.abs, ascending=False).head(8)
    st.bar_chart(contrib.rename("contribution (→ malignant)"))

with st.expander("About the model"):
    st.write(f"L1 regularisation keeps {int((coefs != 0).sum())} of 30 features. "
             "Positive contributions push towards malignant, negative towards benign. "
             "See the notebook for the full model comparison, threshold selection and evaluation.")
