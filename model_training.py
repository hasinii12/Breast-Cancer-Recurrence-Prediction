"""
Breast Cancer Recurrence Prediction - Model Training
=====================================================
This script:
1. Loads the breast cancer dataset
2. Simulates a recurrence label based on medical factors
3. Trains Logistic Regression and Random Forest models
4. Compares their accuracy
5. Saves the best model as model.pkl
"""

import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
import json


def load_and_prepare_data():
    """
    Load the sklearn breast cancer dataset and create features
    that map to our input form fields.

    We simulate recurrence risk based on tumor characteristics,
    since the original dataset predicts malignant/benign.
    """
    # Load the built-in breast cancer dataset
    data = load_breast_cancer()
    df = pd.DataFrame(data.data, columns=data.feature_names)

    # --- Create meaningful features for our prediction form ---

    # Age: simulated between 30-80 (real datasets have similar range)
    np.random.seed(42)
    df['age'] = np.random.randint(30, 80, size=len(df))

    # Tumor Size: use 'mean radius' scaled to mm (typical tumor size 5-50mm)
    df['tumor_size'] = (df['mean radius'] * 2).round(1)

    # Lymph Nodes Involved: simulated 0-15 based on tumor severity
    # More severe tumors (larger, irregular) tend to have more node involvement
    severity = (df['mean radius'] + df['mean compactness'] * 50) / 2
    severity_normalized = (severity - severity.min()) / (severity.max() - severity.min())
    df['lymph_nodes'] = (severity_normalized * 15).round(0).astype(int)

    # Tumor Grade: 1 (low), 2 (medium), 3 (high)
    # Based on texture and compactness
    grade_score = (df['mean texture'] + df['mean compactness'] * 100) / 2
    df['tumor_grade'] = pd.cut(grade_score, bins=3, labels=[1, 2, 3]).astype(int)

    # Hormone Receptor Status: 1 = Positive, 0 = Negative
    # Positive status generally means BETTER prognosis
    df['hormone_receptor'] = (df['mean smoothness'] < df['mean smoothness'].median()).astype(int)

    # HER2 Status: 1 = Positive, 0 = Negative
    # HER2 positive generally means MORE aggressive
    df['her2_status'] = (df['mean symmetry'] > df['mean symmetry'].median()).astype(int)

    # --- Create Recurrence Label ---
    # Recurrence is more likely with:
    #   - Larger tumors, more lymph nodes, higher grade
    #   - HER2 positive, hormone receptor negative
    #   - Original malignant diagnosis (target == 0 in sklearn dataset)
    recurrence_score = (
        0.25 * severity_normalized +                                    # tumor severity
        0.20 * (df['lymph_nodes'] / 15) +                              # lymph node involvement
        0.15 * (df['tumor_grade'] / 3) +                               # tumor grade
        0.15 * df['her2_status'] +                                     # HER2 positive = higher risk
        0.10 * (1 - df['hormone_receptor']) +                          # HR negative = higher risk
        0.10 * (1 - data.target) +                                     # malignant = higher risk
        0.05 * ((df['age'] - 30) / 50)                                 # older age = slightly higher risk
    )

    # Add some randomness to make it realistic
    recurrence_score += np.random.normal(0, 0.08, size=len(df))

    # Use threshold to create binary label (roughly 35% recurrence rate)
    threshold = np.percentile(recurrence_score, 65)
    df['recurrence'] = (recurrence_score >= threshold).astype(int)

    # Select only the features we need for our model
    feature_columns = ['age', 'tumor_size', 'lymph_nodes', 'tumor_grade',
                       'hormone_receptor', 'her2_status']

    X = df[feature_columns]
    y = df['recurrence']

    # Save dataset to CSV for reference
    dataset_df = df[feature_columns + ['recurrence']].copy()
    dataset_df.columns = ['Age', 'Tumor_Size_mm', 'Lymph_Nodes', 'Tumor_Grade',
                          'Hormone_Receptor', 'HER2_Status', 'Recurrence']
    dataset_df.to_csv('dataset/data.csv', index=False)
    print(f"Dataset saved to dataset/data.csv ({len(dataset_df)} samples)")
    print(f"Recurrence distribution:\n{y.value_counts()}\n")

    return X, y, feature_columns


def train_and_compare_models(X_train, X_test, y_train, y_test):
    """
    Train both Logistic Regression and Random Forest.
    Compare their accuracy and return the best one.
    """
    # --- Scale features for Logistic Regression ---
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # ==========================================
    # Model 1: Logistic Regression
    # ==========================================
    print("=" * 50)
    print("Training Logistic Regression...")
    print("=" * 50)

    lr_model = LogisticRegression(random_state=42, max_iter=1000)
    lr_model.fit(X_train_scaled, y_train)
    lr_predictions = lr_model.predict(X_test_scaled)
    lr_accuracy = accuracy_score(y_test, lr_predictions)

    print(f"Accuracy: {lr_accuracy:.4f} ({lr_accuracy*100:.2f}%)")
    print(f"\nClassification Report:\n{classification_report(y_test, lr_predictions)}")

    # ==========================================
    # Model 2: Random Forest
    # ==========================================
    print("=" * 50)
    print("Training Random Forest...")
    print("=" * 50)

    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)  # Random Forest doesn't need scaling
    rf_predictions = rf_model.predict(X_test)
    rf_accuracy = accuracy_score(y_test, rf_predictions)

    print(f"Accuracy: {rf_accuracy:.4f} ({rf_accuracy*100:.2f}%)")
    print(f"\nClassification Report:\n{classification_report(y_test, rf_predictions)}")

    # ==========================================
    # Compare and choose the best model
    # ==========================================
    print("\n" + "=" * 50)
    print("MODEL COMPARISON")
    print("=" * 50)
    print(f"Logistic Regression Accuracy: {lr_accuracy*100:.2f}%")
    print(f"Random Forest Accuracy:       {rf_accuracy*100:.2f}%")

    if rf_accuracy >= lr_accuracy:
        print("\n>> Random Forest is the best model! Saving it.")
        best_model = rf_model
        best_name = "Random Forest"
        best_accuracy = rf_accuracy
        best_predictions = rf_predictions
        # Random Forest doesn't need scaler, but we save it for consistency
        best_scaler = None
    else:
        print("\n>> Logistic Regression is the best model! Saving it.")
        best_model = lr_model
        best_name = "Logistic Regression"
        best_accuracy = lr_accuracy
        best_predictions = lr_predictions
        best_scaler = scaler

    return best_model, best_scaler, best_name, best_accuracy, best_predictions


def plot_confusion_matrix(y_test, predictions, model_name):
    """Create and save confusion matrix plot."""
    cm = confusion_matrix(y_test, predictions)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['No Recurrence', 'Recurrence'],
                yticklabels=['No Recurrence', 'Recurrence'])
    plt.title(f'Confusion Matrix - {model_name}', fontsize=14)
    plt.xlabel('Predicted', fontsize=12)
    plt.ylabel('Actual', fontsize=12)
    plt.tight_layout()
    plt.savefig('static/confusion_matrix.png', dpi=100)
    plt.close()
    print("Confusion matrix saved to static/confusion_matrix.png")


def plot_feature_importance(model, feature_names, model_name):
    """Create and save feature importance plot."""
    if model_name == "Random Forest":
        # Random Forest has built-in feature importance
        importances = model.feature_importances_
    else:
        # For Logistic Regression, use absolute coefficient values
        importances = np.abs(model.coef_[0])

    # Sort features by importance
    indices = np.argsort(importances)
    sorted_features = [feature_names[i] for i in indices]
    sorted_importances = importances[indices]

    # Readable labels for the plot
    label_map = {
        'age': 'Age',
        'tumor_size': 'Tumor Size',
        'lymph_nodes': 'Lymph Nodes',
        'tumor_grade': 'Tumor Grade',
        'hormone_receptor': 'Hormone Receptor',
        'her2_status': 'HER2 Status'
    }
    display_labels = [label_map.get(f, f) for f in sorted_features]

    plt.figure(figsize=(10, 6))
    colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(sorted_features)))
    plt.barh(display_labels, sorted_importances, color=colors)
    plt.xlabel('Importance', fontsize=12)
    plt.title(f'Feature Importance - {model_name}', fontsize=14)
    plt.tight_layout()
    plt.savefig('static/feature_importance.png', dpi=100)
    plt.close()
    print("Feature importance plot saved to static/feature_importance.png")


def plot_model_comparison(lr_accuracy, rf_accuracy):
    """Create and save model comparison bar chart."""
    models = ['Logistic\nRegression', 'Random\nForest']
    accuracies = [lr_accuracy * 100, rf_accuracy * 100]
    colors = ['#3498db', '#2ecc71']

    plt.figure(figsize=(8, 5))
    bars = plt.bar(models, accuracies, color=colors, width=0.5, edgecolor='white')

    # Add accuracy labels on top of bars
    for bar, acc in zip(bars, accuracies):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                 f'{acc:.2f}%', ha='center', va='bottom', fontsize=14, fontweight='bold')

    plt.ylim(0, 105)
    plt.ylabel('Accuracy (%)', fontsize=12)
    plt.title('Model Accuracy Comparison', fontsize=14)
    plt.tight_layout()
    plt.savefig('static/model_comparison.png', dpi=100)
    plt.close()
    print("Model comparison plot saved to static/model_comparison.png")


# ==========================================
# MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    print("=" * 50)
    print("BREAST CANCER RECURRENCE PREDICTION")
    print("Model Training Script")
    print("=" * 50)
    print()

    # Step 1: Load and prepare data
    print("Step 1: Loading and preparing data...")
    X, y, feature_columns = load_and_prepare_data()

    # Step 2: Split into training and testing sets (80/20 split)
    print("Step 2: Splitting data (80% train, 20% test)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Training samples: {len(X_train)}, Testing samples: {len(X_test)}\n")

    # Step 3: Train and compare models
    print("Step 3: Training models...")
    best_model, best_scaler, best_name, best_accuracy, best_predictions = \
        train_and_compare_models(X_train, X_test, y_train, y_test)

    # Step 4: Generate visualizations
    print("\nStep 4: Generating visualizations...")

    # We need both accuracies for the comparison plot
    scaler_temp = StandardScaler()
    X_train_scaled = scaler_temp.fit_transform(X_train)
    X_test_scaled = scaler_temp.transform(X_test)
    lr_acc = accuracy_score(y_test, LogisticRegression(random_state=42, max_iter=1000)
                            .fit(X_train_scaled, y_train).predict(X_test_scaled))
    rf_acc = accuracy_score(y_test, RandomForestClassifier(n_estimators=100, random_state=42)
                            .fit(X_train, y_train).predict(X_test))

    plot_confusion_matrix(y_test, best_predictions, best_name)
    plot_feature_importance(best_model, feature_columns, best_name)
    plot_model_comparison(lr_acc, rf_acc)

    # Step 5: Save the best model and metadata
    print("\nStep 5: Saving model...")
    model_data = {
        'model': best_model,
        'scaler': best_scaler,
        'feature_columns': feature_columns,
        'model_name': best_name,
        'accuracy': best_accuracy
    }
    joblib.dump(model_data, 'model.pkl')
    print(f"Model saved as model.pkl")

    # Save model info as JSON for the web app
    model_info = {
        'model_name': best_name,
        'accuracy': round(best_accuracy * 100, 2),
        'lr_accuracy': round(lr_acc * 100, 2),
        'rf_accuracy': round(rf_acc * 100, 2),
        'training_samples': len(X_train),
        'testing_samples': len(X_test)
    }
    with open('model_info.json', 'w') as f:
        json.dump(model_info, f, indent=2)
    print("Model info saved to model_info.json")

    print("\n" + "=" * 50)
    print("TRAINING COMPLETE!")
    print(f"Best Model: {best_name}")
    print(f"Accuracy: {best_accuracy*100:.2f}%")
    print("=" * 50)
    print("\nYou can now run the web app with: python app.py")
