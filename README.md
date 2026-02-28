# Breast Cancer Recurrence Prediction

A simple machine learning web application that predicts the risk of breast cancer recurrence using tumor-related data. Built with Python, Flask, and Scikit-learn.

---

## Project Overview

This project uses machine learning to predict whether a breast cancer patient is at **high risk** or **low risk** of cancer recurrence. The user enters patient details through a web form, and the trained model returns a prediction with a confidence probability.

This is a **mini-project** designed to demonstrate:
- Data preprocessing and feature engineering
- Training and comparing ML models
- Building a web interface with Flask
- Displaying results with visualizations

---

## Problem Statement

Breast cancer recurrence is a major concern for patients and doctors. Early prediction of recurrence risk can help in planning better treatment. This project builds a simple predictive model using tumor characteristics to classify recurrence risk as **Low** or **High**.

---

## Dataset Used

- **Source:** Breast Cancer Wisconsin dataset (from `sklearn.datasets`)
- **Samples:** 569
- **Adaptation:** Since the original dataset predicts malignant/benign tumors, we engineered features and simulated a recurrence label based on medical factors like tumor size, lymph node involvement, grade, hormone receptor status, and HER2 status.

### Input Features:
| Feature | Description | Type |
|---------|-------------|------|
| Age | Patient's age in years | Numeric (20-100) |
| Tumor Size | Size of tumor in mm | Numeric (1-100) |
| Lymph Nodes Involved | Number of affected lymph nodes | Numeric (0-30) |
| Tumor Grade | Grade 1 (Low), 2 (Medium), 3 (High) | Categorical |
| Hormone Receptor (ER/PR) | Positive or Negative | Binary |
| HER2 Status | Positive or Negative | Binary |

### Output:
- **Recurrence Risk:** Low or High
- **Probability:** Confidence percentage

---

## Algorithms Used

Two algorithms were trained and compared:

### 1. Logistic Regression
- A simple linear model
- Good for binary classification
- Fast to train
- Requires feature scaling

### 2. Random Forest
- An ensemble of decision trees
- Handles non-linear relationships
- Provides feature importance
- Does not require scaling

The model with **higher accuracy** is automatically selected and saved.

---

## Model Accuracy

| Model | Accuracy |
|-------|----------|
| Logistic Regression | ~85-90% |
| Random Forest | ~90-95% |

> Exact numbers depend on the random split. Run `model_training.py` to see your results.

---

## Project Structure

```
breast-cancer-recurrence/
|
├── app.py                  # Flask web application
├── model_training.py       # Model training script
├── model.pkl               # Saved trained model
├── model_info.json         # Model metadata
├── requirements.txt        # Python dependencies
├── README.md               # This file
│
├── templates/
│   ├── index.html          # Input form page
│   └── result.html         # Prediction result page
│
├── static/
│   ├── style.css           # Custom styles
│   ├── confusion_matrix.png    # Generated plot
│   ├── feature_importance.png  # Generated plot
│   └── model_comparison.png    # Generated plot
│
└── dataset/
    └── data.csv            # Generated dataset
```

---

## How to Run (Step-by-Step)

### Prerequisites
- Python 3.8 or higher installed
- pip (Python package manager)

### Step 1: Clone or Download the Project
```bash
cd breast-cancer-recurrence
```

### Step 2: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 3: Train the Model
```bash
python model_training.py
```
This will:
- Generate the dataset (`dataset/data.csv`)
- Train both models and compare accuracy
- Save the best model as `model.pkl`
- Generate visualization plots in `static/`

### Step 4: Run the Web Application
```bash
python app.py
```

### Step 5: Open in Browser
Go to: **http://127.0.0.1:5000**

### Step 6: Make a Prediction
1. Fill in the patient details in the form
2. Click "Predict Recurrence Risk"
3. View the result with probability, accuracy, and charts

---

## Sample Output

### Home Page (Input Form)
- Clean form with 6 input fields
- Dropdown menus for categorical inputs
- Submit button for prediction

### Result Page
- Prediction: "Low Risk of Recurrence" or "High Risk of Recurrence"
- Confidence probability (e.g., 87.5%)
- Model accuracy display
- Confusion matrix chart
- Feature importance chart
- Model comparison chart

---

## Technologies Used

| Technology | Purpose |
|-----------|---------|
| Python 3 | Programming language |
| Flask | Web framework |
| Scikit-learn | Machine learning |
| Pandas | Data handling |
| NumPy | Numerical operations |
| Matplotlib | Plotting charts |
| Seaborn | Statistical visualizations |
| Bootstrap 5 | Frontend styling |
| HTML/CSS | Web interface |

---

## Future Improvements

1. **Use a real recurrence dataset** - Replace simulated labels with actual clinical recurrence data
2. **Add more features** - Include treatment type, family history, and genetic markers
3. **Try deep learning** - Use neural networks for potentially better accuracy
4. **Add patient history** - Store and track predictions over time with a database
5. **Deploy online** - Host on Heroku or AWS for remote access
6. **Add data upload** - Allow CSV file upload for batch predictions
7. **Cross-validation** - Use k-fold cross-validation for more robust evaluation

---

## Limitations

1. **Simulated labels** - The recurrence label is engineered, not from real clinical data
2. **Small dataset** - Only 569 samples; real-world models need more data
3. **Limited features** - Only 6 input features; clinical models use many more
4. **Not for medical use** - This is an educational project, not a clinical tool
5. **No user authentication** - Anyone can access the application
6. **Static model** - The model doesn't learn from new predictions

---

## License

This project is for educational purposes only. Free to use for college projects and learning.

---

## Authors

College Mini Project - Breast Cancer Recurrence Prediction
