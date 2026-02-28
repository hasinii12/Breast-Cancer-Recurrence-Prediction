"""
Breast Cancer Recurrence Prediction - Flask Web App
====================================================
This is the main web application file.
It loads the trained model and provides a simple web interface
for predicting breast cancer recurrence risk.
"""

from flask import Flask, render_template, request
import joblib
import numpy as np
import json
import os

# Create the Flask app
app = Flask(__name__)

# Load the trained model when the app starts
model_data = joblib.load('model.pkl')
model = model_data['model']
scaler = model_data['scaler']
feature_columns = model_data['feature_columns']
model_name = model_data['model_name']
accuracy = model_data['accuracy']

# Load model comparison info
with open('model_info.json', 'r') as f:
    model_info = json.load(f)


@app.route('/')
def home():
    """Display the input form (home page)."""
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    """
    Handle form submission:
    1. Get input values from the form
    2. Prepare the data for the model
    3. Make a prediction
    4. Show the result page
    """
    # Get values from the form
    age = float(request.form['age'])
    tumor_size = float(request.form['tumor_size'])
    lymph_nodes = int(request.form['lymph_nodes'])
    tumor_grade = int(request.form['tumor_grade'])
    hormone_receptor = int(request.form['hormone_receptor'])
    her2_status = int(request.form['her2_status'])

    # Arrange input as array (same order as training features)
    input_data = np.array([[age, tumor_size, lymph_nodes, tumor_grade,
                            hormone_receptor, her2_status]])

    # Scale the input if the model requires it (Logistic Regression)
    if scaler is not None:
        input_data = scaler.transform(input_data)

    # Make prediction
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0]

    # Determine result
    if prediction == 1:
        result = "High Risk of Recurrence"
        risk_class = "high"
        risk_probability = round(probability[1] * 100, 2)
    else:
        result = "Low Risk of Recurrence"
        risk_class = "low"
        risk_probability = round(probability[0] * 100, 2)

    # Pass all data to the result template
    return render_template('result.html',
                           result=result,
                           risk_class=risk_class,
                           probability=risk_probability,
                           model_name=model_name,
                           accuracy=round(accuracy * 100, 2),
                           model_info=model_info,
                           # Pass back input values for display
                           age=int(age),
                           tumor_size=tumor_size,
                           lymph_nodes=lymph_nodes,
                           tumor_grade=tumor_grade,
                           hormone_receptor=hormone_receptor,
                           her2_status=her2_status)


@app.route('/about')
def about():
    """Simple about page with model details."""
    return render_template('index.html', show_about=True, model_info=model_info)


# Run the app
if __name__ == '__main__':
    print("=" * 50)
    print("Breast Cancer Recurrence Prediction App")
    print(f"Model: {model_name} (Accuracy: {accuracy*100:.2f}%)")
    print("=" * 50)
    print("Open http://127.0.0.1:5000 in your browser")
    print("Press Ctrl+C to stop the server")
    print("=" * 50)
    app.run(debug=True)
