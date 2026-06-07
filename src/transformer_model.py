import os
import pandas as pd
import numpy as np
import torch
import joblib
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments

# Directories
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
PROCESSED_DATA_DIR = os.path.join(BASE_DIR, 'data', 'processed')
MODELS_DIR = os.path.join(BASE_DIR, 'models')
TRANSFORMER_PATH = os.path.join(MODELS_DIR, 'transformer_checkpoint')

MODEL_NAME = 'prajjwal1/bert-tiny'  # Extremely lightweight BERT (4.4M params), perfect for CPU

class MedicalDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

    def __len__(self):
        return len(self.labels)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='weighted', zero_division=0)
    return {
        'accuracy': acc,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

def train_transformer():
    os.makedirs(TRANSFORMER_PATH, exist_ok=True)
    
    train_text_path = os.path.join(PROCESSED_DATA_DIR, 'train_text.csv')
    test_text_path = os.path.join(PROCESSED_DATA_DIR, 'test_text.csv')
    le_path = os.path.join(MODELS_DIR, 'label_encoder.joblib')
    
    if not os.path.exists(train_text_path) or not os.path.exists(test_text_path):
        print("Processed text datasets not found. Run data_processor.py first.")
        return
        
    if not os.path.exists(le_path):
        print("Label Encoder not found. Run ml_models.py first to establish the baseline encoder.")
        return
        
    df_train = pd.read_csv(train_text_path)
    df_test = pd.read_csv(test_text_path)
    
    # Load the shared LabelEncoder
    le = joblib.load(le_path)
    
    # Encode labels
    train_labels = le.transform(df_train['label'].str.strip())
    test_labels = le.transform(df_test['label'].str.strip())
    
    num_classes = len(le.classes_)
    print(f"Loaded {num_classes} classes from label encoder.")
    
    # Load BERT Tokenizer
    print(f"Loading tokenizer {MODEL_NAME}...")
    tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
    
    # Tokenize datasets
    print("Tokenizing train and test datasets...")
    train_encodings = tokenizer(list(df_train['text']), truncation=True, padding=True, max_length=64)
    test_encodings = tokenizer(list(df_test['text']), truncation=True, padding=True, max_length=64)
    
    train_dataset = MedicalDataset(train_encodings, train_labels)
    test_dataset = MedicalDataset(test_encodings, test_labels)
    
    # Load BERT Sequence Classification Model
    print(f"Initializing {MODEL_NAME} for {num_classes} classes...")
    model = BertForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=num_classes)
    
    # Move model to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    print(f"Model loaded onto device: {device}")
    
    # Define training arguments
    training_args = TrainingArguments(
        output_dir=os.path.join(MODELS_DIR, 'bert_results'),
        num_train_epochs=5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        warmup_ratio=0.1,
        weight_decay=0.01,
        logging_dir=os.path.join(MODELS_DIR, 'bert_logs'),
        logging_steps=100,
        eval_strategy='epoch',
        save_strategy='epoch',
        learning_rate=5e-5,
        load_best_model_at_end=True,
        metric_for_best_model='accuracy',
        report_to='none'  # Disable integrations (wandb, etc.)
    )
    
    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics
    )
    
    print("--- Training BERT Transformer Classifier ---")
    trainer.train()
    
    # Evaluate model
    print("Evaluating transformer model on test set...")
    eval_results = trainer.evaluate()
    print(f"Transformer Evaluation -> Accuracy: {eval_results['eval_accuracy']:.4f}, F1-Score: {eval_results['eval_f1']:.4f}")
    
    # Save the best model and tokenizer
    model.save_pretrained(TRANSFORMER_PATH)
    tokenizer.save_pretrained(TRANSFORMER_PATH)
    print(f"Saved best transformer checkpoint and tokenizer to {TRANSFORMER_PATH}")
    
    # Append transformer results to the evaluation report
    report_path = os.path.join(BASE_DIR, 'artifacts', 'model_evaluation_report.md')
    if os.path.exists(report_path):
        with open(report_path, 'r') as f:
            content = f.read()
            
        with open(report_path, 'w') as f:
            f.write(content)
            f.write("\n\n## Transformer-Based Model (Natural Language Descriptions)\n\n")
            f.write("| Model Name | Test Accuracy | Test Precision (Weighted) | Test Recall (Weighted) | Test F1-Score (Weighted) |\n")
            f.write("| --- | --- | --- | --- | --- |\n")
            f.write(f"| BERT-Tiny ({MODEL_NAME}) | {eval_results['eval_accuracy']:.4f} | {eval_results['eval_precision']:.4f} | {eval_results['eval_recall']:.4f} | {eval_results['eval_f1']:.4f} |\n")
            
        print("Updated artifacts/model_evaluation_report.md with Transformer metrics.")

if __name__ == '__main__':
    train_transformer()
