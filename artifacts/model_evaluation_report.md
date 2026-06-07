# Model Evaluation Report

This report summarizes the performance metrics of the trained medical diagnosis classifiers.

## Traditional Machine Learning Models (Binary Symptom Vectors)

| Model Name | Accuracy | Precision (Weighted) | Recall (Weighted) | F1-Score (Weighted) |
| --- | --- | --- | --- | --- |
| Logistic Regression | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| Random Forest | 0.9762 | 0.9881 | 0.9762 | 0.9762 |
| Naive Bayes | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| Support Vector Machine | 1.0000 | 1.0000 | 1.0000 | 1.0000 |


*Note: High scores are expected on clean test inputs because symptoms align precisely with disease definitions in the structured dataset.*

## Transformer-Based Model (Natural Language Descriptions)

| Model Name | Test Accuracy | Test Precision (Weighted) | Test Recall (Weighted) | Test F1-Score (Weighted) |
| --- | --- | --- | --- | --- |
| BERT-Tiny (prajjwal1/bert-tiny) | 0.9524 | 0.9524 | 0.9524 | 0.9444 |
