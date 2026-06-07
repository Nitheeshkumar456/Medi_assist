import os
import pandas as pd
import numpy as np
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report

# Directories
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
RAW_DATA_DIR = os.path.join(BASE_DIR, 'data', 'raw')
MODELS_DIR = os.path.join(BASE_DIR, 'models')
PLOTS_DIR = os.path.join(BASE_DIR, 'artifacts', 'plots')

def train_traditional_models():
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(PLOTS_DIR, exist_ok=True)
    
    train_path = os.path.join(RAW_DATA_DIR, 'training_data.csv')
    test_path = os.path.join(RAW_DATA_DIR, 'test_data.csv')
    
    if not os.path.exists(train_path) or not os.path.exists(test_path):
        print("Data files not found. Run data_processor.py first.")
        return
        
    # Load binary dataset
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    
    # Strip columns in case of trailing spaces
    train_df.columns = [c.strip() for c in train_df.columns]
    test_df.columns = [c.strip() for c in test_df.columns]
    
    # Filter symptoms to keep only columns present in both and not unnamed or target prognosis
    symptom_cols = [c for c in train_df.columns if c in test_df.columns and c != 'prognosis' and not c.startswith('Unnamed:')]
    
    X_train = train_df[symptom_cols].fillna(0)
    y_train = train_df['prognosis'].str.strip()
    
    X_test = test_df[symptom_cols].fillna(0)
    y_test = test_df['prognosis'].str.strip()

    
    # Fit label encoder
    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train)
    y_test_enc = le.transform(y_test)
    
    # Save label encoder
    joblib.dump(le, os.path.join(MODELS_DIR, 'label_encoder.joblib'))
    
    # Initialize traditional models
    models = {
        'Logistic Regression': LogisticRegression(max_iter=500, random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'Naive Bayes': MultinomialNB(),
        'Support Vector Machine': SVC(kernel='linear', probability=True, random_state=42)
    }
    
    results = {}
    trained_models = {}
    
    print("--- Training Traditional ML Models ---")
    for name, clf in models.items():
        print(f"Training {name}...")
        clf.fit(X_train, y_train_enc)
        preds = clf.predict(X_test)
        
        # Calculate metrics
        acc = accuracy_score(y_test_enc, preds)
        precision, recall, f1, _ = precision_recall_fscore_support(y_test_enc, preds, average='weighted', zero_division=0)
        
        print(f"{name} Metrics -> Accuracy: {acc:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1-Score: {f1:.4f}\n")
        
        results[name] = {
            'Accuracy': acc,
            'Precision': precision,
            'Recall': recall,
            'F1': f1
        }
        trained_models[name] = clf
        
    # Save the models dictionary
    joblib.dump(trained_models, os.path.join(MODELS_DIR, 'traditional_models.joblib'))
    print("Traditional models serialized to models/traditional_models.joblib")
    
    # Generate visual comparison plot
    metrics_df = pd.DataFrame(results).T
    
    ax = metrics_df.plot(kind='bar', figsize=(10, 6), colormap='viridis')
    plt.title('Comparison of Traditional Machine Learning Models')
    plt.xlabel('Classifier')
    plt.ylabel('Score')
    plt.ylim(0, 1.1)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.legend(loc='lower left')
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'model_comparison.png'))
    plt.close()
    print("Generated model comparison plot at artifacts/plots/model_comparison.png")
    
    # Write a detailed evaluation markdown report
    report_path = os.path.join(BASE_DIR, 'artifacts', 'model_evaluation_report.md')
    with open(report_path, 'w') as f:
        f.write("# Model Evaluation Report\n\n")
        f.write("This report summarizes the performance metrics of the trained medical diagnosis classifiers.\n\n")
        f.write("## Traditional Machine Learning Models (Binary Symptom Vectors)\n\n")
        f.write("| Model Name | Accuracy | Precision (Weighted) | Recall (Weighted) | F1-Score (Weighted) |\n")
        f.write("| --- | --- | --- | --- | --- |\n")
        for name, m in results.items():
            f.write(f"| {name} | {m['Accuracy']:.4f} | {m['Precision']:.4f} | {m['Recall']:.4f} | {m['F1']:.4f} |\n")
        f.write("\n\n*Note: High scores are expected on clean test inputs because symptoms align precisely with disease definitions in the structured dataset.*")
        
    print(f"Saved text report to {report_path}")

if __name__ == '__main__':
    train_traditional_models()
