from flask import Flask, render_template, request, jsonify
import joblib
import pandas as pd
import os

app = Flask(__name__, template_folder='Template')

base_dir = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(base_dir, "rf_heart_model.pkl")

if os.path.exists(MODEL_PATH):
    rf_model = joblib.load(MODEL_PATH)
    print("SUCCESS: Model loaded!")
else:
    rf_model = None
    print("ERROR: Run train_model.py first!")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/diagnosis')
def diagnosis_page():
    return render_template('diagnosis.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')

@app.route("/predict", methods=["POST"])
def predict():
    if rf_model is None:
        return jsonify({"error": "Model missing. Run train_model.py first."}), 500
    try:
        data = request.get_json()
        feature_names = ['age','sex','cp','trestbps','chol','fbs',
                        'restecg','thalach','exang','oldpeak','slope','ca','thal']
        input_data = [float(data.get(name, 0)) for name in feature_names]
        df = pd.DataFrame([input_data], columns=feature_names)
        prediction = rf_model.predict(df)[0]
        probability = rf_model.predict_proba(df)[0][1]
        result_text = "Heart Disease Detected" if int(prediction) == 1 else "No Heart Disease Detected"
        return jsonify({"prediction": result_text, "probability": float(probability)})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run(debug=True)