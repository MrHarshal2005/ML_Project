"""
Classification with feature selection using ONLY Random Forest.
Loads data from adult.xlsx.
Compares full-feature vs RFE(8) feature-selected model.
Run:  python train_model.py
Output: income_model.pkl
"""
import pandas as pd
import numpy as np
import time
import joblib
import os
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif, RFE
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score
import warnings
warnings.filterwarnings('ignore')

# ============ 1. Load dataset from Excel ============



df = pd.read_excel("adult_income_data.xlsx")

# If your Excel file has NO header row, uncomment these two lines:
# COLS = ["age","workclass","fnlwgt","education","education-num",
#         "marital-status","occupation","relationship","race","sex",
#         "capital-gain","capital-loss","hours-per-week","native-country","income"]
# df = pd.read_excel(EXCEL_FILE, names=COLS)

print(f"Dataset shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}")

# Clean missing values
df = df.dropna()

# Convert target to binary (handles '>50K', ' >50K', '>50K.', etc.)
df['income'] = (df['income'].astype(str).str.strip().str.rstrip('.') == '>50K').astype(int)

print(f"Positive class ratio: {df['income'].mean():.2%}")

# ============ 2. Preprocessing ============
df = df.drop(columns=['fnlwgt'])

cat_cols = df.select_dtypes(include='object').columns.tolist()
le_dict = {}
for col in cat_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    le_dict[col] = le

X = df.drop(columns=['income'])
y = df['income']
feature_names = X.columns.tolist()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ============ 3. Baseline: all features ============
print("\n" + "="*70)
print("[BASELINE] Random Forest - all features (5-fold CV)")
print("="*70)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

baseline_clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)

cv_scores_base = cross_val_score(baseline_clf, X_train_scaled, y_train,
                                  cv=cv, scoring='f1', n_jobs=-1)

start = time.time()
baseline_clf.fit(X_train_scaled, y_train)
base_train_time = time.time() - start

y_pred_base = baseline_clf.predict(X_test_scaled)
y_proba_base = baseline_clf.predict_proba(X_test_scaled)[:, 1]

baseline_results = {
    "Random Forest": {
        "n_features": len(feature_names),
        "train_time": round(base_train_time, 4),
        "cv_f1_mean": round(cv_scores_base.mean(), 4),
        "cv_f1_std": round(cv_scores_base.std(), 4),
        "f1": round(f1_score(y_test, y_pred_base), 4),
        "auc": round(roc_auc_score(y_test, y_proba_base), 4),
        "accuracy": round(accuracy_score(y_test, y_pred_base), 4),
    }
}
r = baseline_results["Random Forest"]
print(f"CV-F1: {r['cv_f1_mean']:.4f} +/- {r['cv_f1_std']:.4f} | "
      f"Test F1: {r['f1']:.4f} | AUC: {r['auc']:.4f} | Time: {r['train_time']}s")

# ============ 4. Feature selection ============
print("\n" + "="*70)
print("[FEATURE SELECTION]")
print("="*70)

selector_kbest = SelectKBest(f_classif, k=8)
X_train_kbest = selector_kbest.fit_transform(X_train_scaled, y_train)
X_test_kbest = selector_kbest.transform(X_test_scaled)
kbest_features = [f for f, s in zip(feature_names, selector_kbest.get_support()) if s]
print(f"SelectKBest (8): {kbest_features}")

rfe = RFE(RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1),
          n_features_to_select=8, step=1)
X_train_rfe = rfe.fit_transform(X_train_scaled, y_train)
X_test_rfe = rfe.transform(X_test_scaled)
rfe_features = [f for f, s in zip(feature_names, rfe.support_) if s]
print(f"RFE (8):         {rfe_features}")

# ============ 5. Optimized: RFE-selected features ============
print("\n" + "="*70)
print("[OPTIMIZED] Random Forest - RFE(8) (5-fold CV)")
print("="*70)

opt_clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)

cv_scores_rfe = cross_val_score(opt_clf, X_train_rfe, y_train,
                                 cv=cv, scoring='f1', n_jobs=-1)

start = time.time()
opt_clf.fit(X_train_rfe, y_train)
opt_train_time = time.time() - start

y_pred_rfe = opt_clf.predict(X_test_rfe)
y_proba_rfe = opt_clf.predict_proba(X_test_rfe)[:, 1]

optimized_results = {
    "Random Forest + RFE(8)": {
        "n_features": X_train_rfe.shape[1],
        "train_time": round(opt_train_time, 4),
        "cv_f1_mean": round(cv_scores_rfe.mean(), 4),
        "cv_f1_std": round(cv_scores_rfe.std(), 4),
        "f1": round(f1_score(y_test, y_pred_rfe), 4),
        "auc": round(roc_auc_score(y_test, y_proba_rfe), 4),
        "accuracy": round(accuracy_score(y_test, y_pred_rfe), 4),
    }
}
r = optimized_results["Random Forest + RFE(8)"]
print(f"CV-F1: {r['cv_f1_mean']:.4f} +/- {r['cv_f1_std']:.4f} | "
      f"Test F1: {r['f1']:.4f} | AUC: {r['auc']:.4f} | Time: {r['train_time']}s")

# ============ 6. Save final model (RFE version) ============
final_clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
final_clf.fit(X_train_rfe, y_train)

bundle = {
    "model": final_clf,
    "scaler": scaler,
    "rfe": rfe,
    "feature_names": feature_names,
    "selected_features": rfe_features,
    "le_dict": le_dict,
    "baseline": baseline_results,
    "optimized": optimized_results,
}
joblib.dump(bundle, "income_model.pkl")
print("\nModel saved to income_model.pkl")

# ============ 7. Summary ============
print("\n" + "="*70)
print("[SUMMARY] Full-feature vs RFE(8)")
print("="*70)
b = baseline_results["Random Forest"]
o = optimized_results["Random Forest + RFE(8)"]
saved = (b['train_time'] - o['train_time']) / b['train_time'] * 100
print(f"{'Metric':15s} | {'Full Features':>14s} | {'RFE(8)':>10s} | {'Change':>10s}")
print("-"*60)
print(f"{'# Features':15s} | {b['n_features']:>14d} | {o['n_features']:>10d} | "
      f"{o['n_features']-b['n_features']:>+10d}")
print(f"{'Train time (s)':15s} | {b['train_time']:>14.4f} | {o['train_time']:>10.4f} | "
      f"{-saved:>9.1f}%")
print(f"{'CV F1':15s} | {b['cv_f1_mean']:>14.4f} | {o['cv_f1_mean']:>10.4f} | "
      f"{o['cv_f1_mean']-b['cv_f1_mean']:>+10.4f}")
print(f"{'Test F1':15s} | {b['f1']:>14.4f} | {o['f1']:>10.4f} | "
      f"{o['f1']-b['f1']:>+10.4f}")
print(f"{'AUC-ROC':15s} | {b['auc']:>14.4f} | {o['auc']:>10.4f} | "
      f"{o['auc']-b['auc']:>+10.4f}")
print(f"{'Accuracy':15s} | {b['accuracy']:>14.4f} | {o['accuracy']:>10.4f} | "
      f"{o['accuracy']-b['accuracy']:>+10.4f}")
