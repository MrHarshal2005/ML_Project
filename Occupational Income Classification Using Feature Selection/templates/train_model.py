"""Train and compare a full-feature model with a compact RFE model.

Feature count and Random Forest settings are selected using stratified
cross-validation on the training split. RFE is refit inside each fold to avoid
using validation-fold labels during feature selection.
"""

import time
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

warnings.filterwarnings("ignore")


DATA_PATH = "Income_Dataset_India.xlsx"
MODEL_PATH = "income_model_india.pkl"
RFE_TREES = 60
MODEL_TREES = 200
PARAMETER_OPTIONS = [
    {"max_features": "sqrt", "min_samples_leaf": 1},
    {"max_features": "sqrt", "min_samples_leaf": 2},
    {"max_features": 0.8, "min_samples_leaf": 1},
    {"max_features": 0.8, "min_samples_leaf": 2},
]


def make_forest(params, n_estimators=MODEL_TREES):
    return RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=42,
        n_jobs=-1,
        **params,
    )


def multiclass_auc(classifier, features, targets):
    probabilities = classifier.predict_proba(features)
    return roc_auc_score(targets, probabilities, multi_class="ovr", average="weighted")


def baseline_clf_cv_predict(x_data, y_data, train_idx, valid_idx, params):
    """Fit a full-feature forest for one fold and return validation predictions."""
    classifier = make_forest(params)
    classifier.fit(x_data[train_idx], y_data[train_idx])
    return classifier.predict(x_data[valid_idx])


# Load and prepare the India income dataset.
df = pd.read_excel(DATA_PATH).dropna()
df = df.drop(columns=["No.", "Income (INR)"], errors="ignore")
print(f"Dataset shape: {df.shape}")
print(f"Class distribution:\n{df['Income Category'].value_counts()}")

target_le = LabelEncoder()
y = target_le.fit_transform(df.pop("Income Category"))
class_names = target_le.classes_.tolist()

le_dict = {}
for column in df.select_dtypes(include="object").columns:
    encoder = LabelEncoder()
    df[column] = encoder.fit_transform(df[column].astype(str))
    le_dict[column] = encoder

X = df
feature_names = X.columns.tolist()
n_features = len(feature_names)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
folds = list(cv.split(X_train, y_train))


# Search every possible feature count. Each fold learns its own RFE ranking,
# and then compares several forest settings by accuracy.
print("\n[SEARCH] Cross-validating feature counts and forest settings")
feature_search = []
for k in range(1, n_features + 1):
    scores_by_params = {i: [] for i in range(len(PARAMETER_OPTIONS))}
    for fold_number, (train_idx, valid_idx) in enumerate(folds):
        x_fold_train = X_train.iloc[train_idx]
        x_fold_valid = X_train.iloc[valid_idx]
        y_fold_train = y_train[train_idx]
        y_fold_valid = y_train[valid_idx]

        selector = RFE(
            make_forest({"max_features": "sqrt", "min_samples_leaf": 1}, RFE_TREES),
            n_features_to_select=k,
            step=1,
        )
        x_fold_train_selected = selector.fit_transform(x_fold_train, y_fold_train)
        x_fold_valid_selected = selector.transform(x_fold_valid)

        for param_index, params in enumerate(PARAMETER_OPTIONS):
            classifier = make_forest(params)
            classifier.fit(x_fold_train_selected, y_fold_train)
            scores_by_params[param_index].append(
                accuracy_score(y_fold_valid, classifier.predict(x_fold_valid_selected))
            )

    parameter_results = []
    for param_index, scores in scores_by_params.items():
        parameter_results.append({
            "params": PARAMETER_OPTIONS[param_index],
            "scores": scores,
            "mean": float(np.mean(scores)),
            "std": float(np.std(scores, ddof=1)),
        })
    best_for_k = max(parameter_results, key=lambda result: result["mean"])
    feature_search.append({
        "k": k,
        "cv_accuracy_mean": round(best_for_k["mean"], 4),
        "cv_accuracy_std": round(best_for_k["std"], 4),
        "params": best_for_k["params"],
    })
    print(
        f"RFE({k:2d}) CV accuracy: {best_for_k['mean']:.4f} "
        f"+/- {best_for_k['std']:.4f} | {best_for_k['params']}"
    )

# Use the one-standard-error rule: retain the smallest feature count whose
# CV accuracy is statistically close to the best observed mean.
best_cv_result = max(feature_search, key=lambda result: result["cv_accuracy_mean"])
accuracy_floor = best_cv_result["cv_accuracy_mean"] - (
    best_cv_result["cv_accuracy_std"] / np.sqrt(len(folds))
)
eligible = [row for row in feature_search if row["cv_accuracy_mean"] >= accuracy_floor]
best_k = min(row["k"] for row in eligible)
chosen_search_row = next(row for row in feature_search if row["k"] == best_k)
chosen_params = chosen_search_row["params"]
print(
    f"\nSelected RFE({best_k}) using the one-standard-error rule; "
    f"CV accuracy={chosen_search_row['cv_accuracy_mean']:.4f}, "
    f"best observed={best_cv_result['cv_accuracy_mean']:.4f}."
)


# Fit preprocessing, selectors, and classifiers on the training split only.
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

baseline_clf = make_forest(chosen_params)
start = time.time()
baseline_clf.fit(X_train_scaled, y_train)
baseline_train_time = time.time() - start
baseline_pred = baseline_clf.predict(X_test_scaled)
baseline_cv_accuracy = [
    accuracy_score(y_train[valid_idx], baseline_clf_cv_predict(
        X_train_scaled, y_train, train_idx, valid_idx, chosen_params
    ))
    for train_idx, valid_idx in folds
]
baseline_results = {
    "Random Forest": {
        "n_features": n_features,
        "train_time": round(baseline_train_time, 4),
        "cv_accuracy_mean": round(float(np.mean(baseline_cv_accuracy)), 4),
        "cv_accuracy_std": round(float(np.std(baseline_cv_accuracy, ddof=1)), 4),
        "f1": round(f1_score(y_test, baseline_pred, average="weighted"), 4),
        "auc": round(multiclass_auc(baseline_clf, X_test_scaled, y_test), 4),
        "accuracy": round(accuracy_score(y_test, baseline_pred), 4),
    }
}

rfe = RFE(
    make_forest({"max_features": "sqrt", "min_samples_leaf": 1}, RFE_TREES),
    n_features_to_select=best_k,
    step=1,
)
X_train_selected = rfe.fit_transform(X_train_scaled, y_train)
X_test_selected = rfe.transform(X_test_scaled)
selected_features = [
    name for name, selected in zip(feature_names, rfe.support_) if selected
]

optimized_clf = make_forest(chosen_params)
start = time.time()
optimized_clf.fit(X_train_selected, y_train)
optimized_train_time = time.time() - start
optimized_pred = optimized_clf.predict(X_test_selected)
opt_label = f"Random Forest + RFE({best_k})"
optimized_results = {
    opt_label: {
        "n_features": len(selected_features),
        "train_time": round(optimized_train_time, 4),
        "cv_accuracy_mean": chosen_search_row["cv_accuracy_mean"],
        "cv_accuracy_std": chosen_search_row["cv_accuracy_std"],
        "f1": round(f1_score(y_test, optimized_pred, average="weighted"), 4),
        "auc": round(multiclass_auc(optimized_clf, X_test_selected, y_test), 4),
        "accuracy": round(accuracy_score(y_test, optimized_pred), 4),
    }
}

bundle = {
    "model": optimized_clf,
    "scaler": scaler,
    "rfe": rfe,
    "feature_names": feature_names,
    "selected_features": selected_features,
    "best_k": best_k,
    "le_dict": le_dict,
    "target_le": target_le,
    "class_names": class_names,
    "baseline": baseline_results,
    "optimized": optimized_results,
    "opt_label": opt_label,
    "feature_search": feature_search,
    "selection_metric": "accuracy",
    "selection_rule": "one_standard_error",
    "model_params": chosen_params,
}
joblib.dump(bundle, MODEL_PATH)
print(f"\nModel saved to {MODEL_PATH}")
print(f"Selected features ({len(selected_features)}): {selected_features}")
print(
    f"Holdout accuracy: baseline={baseline_results['Random Forest']['accuracy']:.4f}, "
    f"RFE={optimized_results[opt_label]['accuracy']:.4f}"
)
