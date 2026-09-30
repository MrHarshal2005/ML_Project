from flask import Flask, render_template, request
import joblib
import pandas as pd

app = Flask(__name__)

print("Loading model...")
bundle = joblib.load("income_model.pkl")
print("Model loaded.")

model = bundle["model"]
scaler = bundle["scaler"]
rfe = bundle["rfe"]
feature_names = bundle["feature_names"]
le_dict = bundle["le_dict"]
baseline = bundle["baseline"]
optimized = bundle["optimized"]
selected_features = bundle["selected_features"]


FORM_FIELDS = [
    {"name": "age", "label": "Age", "type": "number", "default": 35, "min": 17, "max": 90},
    {"name": "workclass", "label": "Workclass", "type": "select",
     "options": ["Private", "Self-emp-not-inc", "Self-emp-inc", "Federal-gov",
                 "Local-gov", "State-gov", "Without-pay", "Never-worked"]},
    {"name": "education", "label": "Education", "type": "select",
     "options": ["Bachelors", "Some-college", "11th", "HS-grad", "Prof-school",
                 "Assoc-acdm", "Assoc-voc", "9th", "7th-8th", "12th",
                 "Masters", "1st-4th", "10th", "Doctorate", "5th-6th",
                 "Preschool"]},
    {"name": "education-num", "label": "Education Num", "type": "number",
     "default": 13, "min": 1, "max": 16},
    {"name": "marital-status", "label": "Marital Status", "type": "select",
     "options": ["Married-civ-spouse", "Divorced", "Never-married", "Separated",
                 "Widowed", "Married-spouse-absent", "Married-AF-spouse"]},
    {"name": "occupation", "label": "Occupation", "type": "select",
     "options": ["Tech-support", "Craft-repair", "Other-service", "Sales",
                 "Exec-managerial", "Prof-specialty", "Handlers-cleaners",
                 "Machine-op-inspct", "Adm-clerical", "Farming-fishing",
                 "Transport-moving", "Priv-house-serv", "Protective-serv",
                 "Armed-Forces"]},
    {"name": "relationship", "label": "Relationship", "type": "select",
     "options": ["Wife", "Own-child", "Husband", "Not-in-family",
                 "Other-relative", "Unmarried"]},
    {"name": "race", "label": "Race", "type": "select",
     "options": ["White", "Black", "Asian-Pac-Islander",
                 "Amer-Indian-Eskimo", "Other"]},
    {"name": "sex", "label": "Sex", "type": "select",
     "options": ["Male", "Female"]},
    {"name": "capital-gain", "label": "Capital Gain", "type": "number",
     "default": 0, "min": 0, "max": 100000},
    {"name": "capital-loss", "label": "Capital Loss", "type": "number",
     "default": 0, "min": 0, "max": 5000},
    {"name": "hours-per-week", "label": "Hours per Week", "type": "number",
     "default": 40, "min": 1, "max": 99},
    {"name": "native-country", "label": "Native Country", "type": "select",
     "options": ["United-States", "Mexico", "Philippines", "Germany", "Canada",
                 "India", "England", "Cuba", "China", "Japan", "Italy",
                 "Dominican-Republic", "Vietnam", "Guatemala", "Iran",
                 "Poland", "Ireland", "France", "Taiwan", "Haiti", "Other"]},
]


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html", fields=FORM_FIELDS)


@app.route("/predict", methods=["POST"])
def predict():
    input_data = {}
    for f in FORM_FIELDS:
        val = request.form.get(f["name"], f.get("default", 0))
        if f["type"] == "number":
            try:
                input_data[f["name"]] = float(val)
            except ValueError:
                input_data[f["name"]] = float(f.get("default", 0))
        else:
            le = le_dict.get(f["name"])
            if le is not None:
                try:
                    input_data[f["name"]] = int(le.transform([val])[0])
                except ValueError:
                    input_data[f["name"]] = 0
            else:
                input_data[f["name"]] = 0

    row = pd.DataFrame([input_data])[feature_names]
    row_scaled = scaler.transform(row)
    row_selected = rfe.transform(row_scaled)

    proba = float(model.predict_proba(row_selected)[0][1])
    is_high = proba > 0.5
    prediction = "> Rs 40,00,000 / year" if is_high else "<= Rs 40,00,000 / year"
    confidence = round(proba * 100, 1) if is_high else round((1 - proba) * 100, 1)

    b = baseline["Random Forest"]
    o = optimized["Random Forest + RFE(8)"]
    time_saved = (b['train_time'] - o['train_time']) / b['train_time'] * 100

    comparison = [{
        "name": "Random Forest",
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
                           is_high=is_high,
                           comparison=comparison,
                           selected_features=selected_features)


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)