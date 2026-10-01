from pathlib import Path

from flask import Flask, render_template, request
import joblib
import pandas as pd

app = Flask(__name__)

print("Loading model...")
MODEL_PATH = Path(__file__).resolve().parent / "income_model_india.pkl"
try:
    bundle = joblib.load(MODEL_PATH)
except ImportError as exc:
    if "DLL load failed" in str(exc) or "Application Control" in str(exc):
        raise RuntimeError(
            "The saved model could not load because Windows blocked a "
            "scikit-learn native DLL. This project supports Python 3.14; "
            "install the dependencies from requirements.txt in that "
            "environment. If Windows continues to block the DLL, ask your "
            "administrator to allow the scikit-learn installation or use an "
            "approved Python environment."
        ) from exc
    raise
print("Model loaded.")

model = bundle["model"]
scaler = bundle["scaler"]
rfe = bundle["rfe"]
feature_names = bundle["feature_names"]
le_dict = bundle["le_dict"]
target_le = bundle["target_le"]
class_names = bundle["class_names"]
baseline = bundle["baseline"]
optimized = bundle["optimized"]
selected_features = bundle["selected_features"]
best_k = bundle["best_k"]
opt_label = bundle["opt_label"]
feature_search = bundle.get("feature_search", [])


FORM_FIELDS = [
    {"name": "Age", "label": "Age", "type": "number", "default": 30, "min": 18, "max": 65},
    {"name": "Gender", "label": "Gender", "type": "select",
     "options": ["Male", "Female"]},
    {"name": "Occupation", "label": "Occupation", "type": "select",
     "options": sorted(le_dict["Occupation"].classes_.tolist())},
    {"name": "Education", "label": "Education", "type": "select",
     "options": ["Below 10th", "10th Pass", "12th Pass", "Diploma",
                 "Bachelor's Degree", "Master's Degree", "PhD"]},
    {"name": "Workclass", "label": "Workclass", "type": "select",
     "options": ["Private", "Government", "Self-Employed",
                 "Public Sector Undertaking", "MNC", "Startup", "Unemployed"]},
    {"name": "Capital Gain", "label": "Capital Gain (INR)", "type": "number",
     "default": 0, "min": 0, "max": 500000},
    {"name": "Capital Loss", "label": "Capital Loss (INR)", "type": "number",
     "default": 0, "min": 0, "max": 100000},
    {"name": "City", "label": "City", "type": "select",
     "options": sorted(le_dict["City"].classes_.tolist())},
    {"name": "Years of Work Experience", "label": "Years of Work Experience",
     "type": "number", "default": 5, "min": 0, "max": 45},
    {"name": "Industry", "label": "Industry", "type": "select",
     "options": sorted(le_dict["Industry"].classes_.tolist())},
    {"name": "Hours per Week", "label": "Hours per Week", "type": "number",
     "default": 45, "min": 15, "max": 80},
]

# Keep categorical choices in sync with the encoders used during training.
for field in FORM_FIELDS:
    encoder = le_dict.get(field["name"])
    if field["type"] == "select" and encoder is not None:
        field["options"] = encoder.classes_.tolist()


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html", fields=FORM_FIELDS)


@app.route("/predict", methods=["POST"])
def predict():
    input_data = {}
    for f in FORM_FIELDS:
        name = f["name"]
        val = request.form.get(name, "").strip()
        if not val:
            return render_template(
                "index.html", fields=FORM_FIELDS,
                error=f"Please provide a value for {f['label']}.",
            ), 400

        if f["type"] == "number":
            try:
                number = float(val)
            except ValueError:
                return render_template(
                    "index.html", fields=FORM_FIELDS,
                    error=f"{f['label']} must be a number.",
                ), 400

            if not (number == number and abs(number) != float("inf")):
                return render_template(
                    "index.html", fields=FORM_FIELDS,
                    error=f"{f['label']} must be a finite number.",
                ), 400
            if number < f.get("min", number) or number > f.get("max", number):
                return render_template(
                    "index.html", fields=FORM_FIELDS,
                    error=f"{f['label']} must be between {f['min']} and {f['max']}.",
                ), 400
            input_data[name] = number
        else:
            le = le_dict.get(name)
            if le is None or val not in le.classes_:
                return render_template(
                    "index.html", fields=FORM_FIELDS,
                    error=f"{f['label']} has an unsupported value. Please choose an available option.",
                ), 400
            input_data[name] = int(le.transform([val])[0])

    row = pd.DataFrame([input_data])[feature_names]
    row_scaled = scaler.transform(row)
    row_selected = rfe.transform(row_scaled)

    proba = model.predict_proba(row_selected)[0]
    pred_idx = proba.argmax()
    prediction = class_names[pred_idx]
    confidence = round(float(proba[pred_idx]) * 100, 1)

    class_probs = [
        {"label": class_names[i], "prob": round(float(p) * 100, 1)}
        for i, p in enumerate(proba)
    ]

    b = baseline["Random Forest"]
    o = optimized[opt_label]
    time_saved = (
        (b['train_time'] - o['train_time']) / b['train_time'] * 100
        if b['train_time'] else 0
    )

    comparison = [{
        "name": "Random Forest",
        "opt_name": opt_label,
        "base_features": b["n_features"],
        "opt_features": o["n_features"],
        "base_time": b["train_time"],
        "opt_time": o["train_time"],
        "time_saved": round(time_saved, 1),
        "base_f1": b["f1"],
        "opt_f1": o["f1"],
        "f1_delta": round(o["f1"] - b["f1"], 4),
        "base_auc": b["auc"],
        "opt_auc": o["auc"],
        "auc_delta": round(o["auc"] - b["auc"], 4),
        "base_acc": b["accuracy"],
        "opt_acc": o["accuracy"],
    }]

    return render_template("result.html",
                           prediction=prediction,
                           confidence=confidence,
                           class_probs=class_probs,
                           comparison=comparison,
                           selected_features=selected_features,
                           best_k=best_k,
                           feature_search=feature_search)


if __name__ == "__main__":
    app.run(debug=False, host="127.0.0.1", port=5000)
