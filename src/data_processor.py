import os
import urllib.request
import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# Local imports
from src.database import init_db, populate_knowledge_base

# Dataset URLs
URL_TRAINING = "https://raw.githubusercontent.com/anujdutt9/Disease-Prediction-from-Symptoms/master/dataset/training_data.csv"
URL_TESTING = "https://raw.githubusercontent.com/anujdutt9/Disease-Prediction-from-Symptoms/master/dataset/test_data.csv"
URL_DESCRIPTION = "https://raw.githubusercontent.com/itachi9604/healthcare-chatbot/master/MasterData/symptom_Description.csv"
URL_PRECAUTION = "https://raw.githubusercontent.com/itachi9604/healthcare-chatbot/master/MasterData/symptom_precaution.csv"
URL_SEVERITY = "https://raw.githubusercontent.com/harshaljagtap6/Disease-Prediction-ML/master/Symptom-severity.csv"

# Local directories
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
RAW_DATA_DIR = os.path.join(BASE_DIR, 'data', 'raw')
PROCESSED_DATA_DIR = os.path.join(BASE_DIR, 'data', 'processed')
PLOTS_DIR = os.path.join(BASE_DIR, 'artifacts', 'plots')

# Specialist mapping based on standard medical expertise for the 41 diseases
SPECIALIST_MAPPING = {
    'Fungal infection': 'Dermatologist',
    'Allergy': 'Allergist / Immunologist',
    'GERD': 'Gastroenterologist',
    'Chronic cholestasis': 'Hepatologist / Gastroenterologist',
    'Drug Reaction': 'Allergist / Dermatologist',
    'Peptic ulcer disease': 'Gastroenterologist',
    'AIDS': 'Infectious Disease Specialist',
    'Diabetes': 'Endocrinologist',
    'Gastroenteritis': 'Gastroenterologist',
    'Bronchial Asthma': 'Pulmonologist / Allergist',
    'Hypertension': 'Cardiologist',
    'Migraine': 'Neurologist',
    'Cervical spondylosis': 'Orthopedic Surgeon / Neurologist',
    'Paralysis (brain hemorrhage)': 'Neurologist / Neurosurgeon',
    'Jaundice': 'Hepatologist / Gastroenterologist',
    'Malaria': 'Infectious Disease Specialist',
    'Chicken pox': 'Pediatrician / Infectious Disease Specialist',
    'Dengue': 'Infectious Disease Specialist',
    'Typhoid': 'Infectious Disease Specialist',
    'Hepatitis A': 'Hepatologist',
    'Hepatitis B': 'Hepatologist',
    'Hepatitis C': 'Hepatologist',
    'Hepatitis D': 'Hepatologist',
    'Hepatitis E': 'Hepatologist',
    'Alcoholic hepatitis': 'Hepatologist / Gastroenterologist',
    'Tuberculosis': 'Pulmonologist / Infectious Disease Specialist',
    'Common Cold': 'General Physician',
    'Pneumonia': 'Pulmonologist',
    'Dimorphic hemmorhoids(piles)': 'Colorectal Surgeon',
    'Heart attack': 'Cardiologist',
    'Varicose veins': 'Vascular Surgeon',
    'Hypothyroidism': 'Endocrinologist',
    'Hyperthyroidism': 'Endocrinologist',
    'Hypoglycemia': 'Endocrinologist',
    'Osteoarthristis': 'Rheumatologist / Orthopedist',
    'Arthritis': 'Rheumatologist',
    '(vertigo) Paroxysmal Positional Vertigo': 'ENT Specialist / Neurologist',
    'Acne': 'Dermatologist',
    'Urinary tract infection': 'Urologist / Nephrologist',
    'Psoriasis': 'Dermatologist',
    'Impetigo': 'Dermatologist'
}

def download_datasets():
    os.makedirs(RAW_DATA_DIR, exist_ok=True)
    os.makedirs(PLOTS_DIR, exist_ok=True)
    
    files = {
        'training_data.csv': URL_TRAINING,
        'test_data.csv': URL_TESTING,
        'symptom_Description.csv': URL_DESCRIPTION,
        'symptom_precaution.csv': URL_PRECAUTION,
        'Symptom-severity.csv': URL_SEVERITY
    }
    
    for filename, url in files.items():
        filepath = os.path.join(RAW_DATA_DIR, filename)
        if not os.path.exists(filepath) or os.path.getsize(filepath) == 0:
            print(f"Downloading {filename} from {url}...")
            try:
                urllib.request.urlretrieve(url, filepath)
                print(f"Saved to {filepath}")
            except Exception as e:
                print(f"Error downloading {filename}: {e}")
                if os.path.exists(filepath):
                    os.remove(filepath)
        else:
            print(f"{filename} already exists locally.")

def clean_symptom_name(sym):
    """Normalize symptom strings from column names (e.g. 'muscle_wasting' -> 'muscle wasting')"""
    if not isinstance(sym, str):
        return ""
    sym = sym.strip().lower()
    sym = sym.replace('_', ' ')
    sym = re.sub(r'\s+', ' ', sym)
    return sym

def clean_disease_name(disease):
    """Normalize disease names by stripping spaces and standardizing casing"""
    if not isinstance(disease, str):
        return ""
    # Strip spaces and normalize capitalization
    disease = disease.strip()
    return disease

# Setup NLTK
def setup_nltk():
    try:
        nltk.data.find('corpora/stopwords')
        nltk.data.find('corpora/wordnet')
        nltk.data.find('tokenizers/punkt')
        nltk.data.find('tokenizers/punkt_tab')
    except LookupError:
        print("Downloading NLTK resources...")
        nltk.download('stopwords')
        nltk.download('wordnet')
        nltk.download('punkt')
        nltk.download('punkt_tab')

setup_nltk()

def preprocess_text(text):
    """Applies text cleaning pipeline to natural language symptom descriptions"""
    if not isinstance(text, str):
        return ""
    # Lowercasing
    text = text.lower()
    # Remove special characters and punctuation (keep letters and spaces)
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    # Tokenization
    tokens = word_tokenize(text)
    # Stopword removal
    stop_words = set(stopwords.words('english'))
    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    cleaned_tokens = [lemmatizer.lemmatize(t) for t in tokens if t not in stop_words and len(t) > 1]
    
    return " ".join(cleaned_tokens)

def run_eda_and_generate_plots():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    train_path = os.path.join(RAW_DATA_DIR, 'training_data.csv')
    if not os.path.exists(train_path):
        print("Training dataset not found, download first.")
        return
        
    df = pd.read_csv(train_path)
    
    # 1. Disease Distribution Plot
    plt.figure(figsize=(12, 6))
    disease_counts = df['prognosis'].value_counts()
    disease_counts.plot(kind='bar', color='indigo')
    plt.title('Disease Prognosis Class Distribution')
    plt.xlabel('Disease')
    plt.ylabel('Frequency')
    plt.xticks(rotation=90, fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'disease_distribution.png'))
    plt.close()
    print("Generated disease distribution plot.")
    
    # 2. Symptom Frequency Plot
    symptom_cols = [c for c in df.columns if c != 'prognosis']
    symptom_sums = df[symptom_cols].sum().sort_values(ascending=False)
    
    plt.figure(figsize=(12, 6))
    top_symptoms = symptom_sums.head(20)
    # Clean the index names for visual display
    top_symptoms.index = [clean_symptom_name(idx) for idx in top_symptoms.index]
    top_symptoms.plot(kind='bar', color='teal')
    plt.title('Top 20 Most Frequent Symptoms in Training Data')
    plt.xlabel('Symptom')
    plt.ylabel('Occurrence Count')
    plt.xticks(rotation=45, ha='right', fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'symptom_frequency.png'))
    plt.close()
    print("Generated symptom frequency plot.")

def load_and_sync_knowledge_base():
    """Reads description, precaution, and severity datasets, merges them, and saves to SQLite"""
    desc_path = os.path.join(RAW_DATA_DIR, 'symptom_Description.csv')
    prec_path = os.path.join(RAW_DATA_DIR, 'symptom_precaution.csv')
    sev_path = os.path.join(RAW_DATA_DIR, 'Symptom-severity.csv')
    
    if not (os.path.exists(desc_path) and os.path.exists(prec_path)):
        print("Required CSV files for knowledge base missing.")
        return
        
    # Load without headers since raw files do not have column headers
    df_desc = pd.read_csv(desc_path, header=None, names=['Disease', 'Description'])
    df_prec = pd.read_csv(prec_path, header=None, names=['Disease', 'Precaution_1', 'Precaution_2', 'Precaution_3', 'Precaution_4'])
    
    # Clean column names and whitespace in descriptions
    df_desc['Disease'] = df_desc['Disease'].apply(clean_disease_name)
    df_desc['Description'] = df_desc['Description'].str.strip()
    
    # Clean precautions
    df_prec['Disease'] = df_prec['Disease'].apply(clean_disease_name)
    
    # Merge precautions columns (Precaution_1, Precaution_2, Precaution_3, Precaution_4)
    precaution_cols = ['Precaution_1', 'Precaution_2', 'Precaution_3', 'Precaution_4']
    df_prec['merged_precautions'] = df_prec[precaution_cols].apply(
        lambda row: ", ".join([str(val).strip() for val in row if pd.notna(val) and str(val).strip() != ""]), axis=1
    )
    
    # Create descriptions dictionary
    descriptions = dict(zip(df_desc['Disease'], df_desc['Description']))
    precautions = dict(zip(df_prec['Disease'], df_prec['merged_precautions']))
    
    # Identify unique diseases from training data
    train_path = os.path.join(RAW_DATA_DIR, 'training_data.csv')
    df_train = pd.read_csv(train_path)
    unique_diseases = df_train['prognosis'].apply(clean_disease_name).unique()
    
    knowledge_base_list = []
    
    # Build a combined database entry for each disease in the classification task
    for disease in unique_diseases:
        desc = descriptions.get(disease, f"A health condition characterized by specific symptoms.")
        prec = precautions.get(disease, "Consult a general physician, keep hydrated, and rest.")
        
        # Specialist recommendation
        # Fallback to general physician if not mapped
        specialist = SPECIALIST_MAPPING.get(disease, "General Physician")
        
        # Check standard severity logic (e.g. Heart attack is High, Common Cold is Low)
        severity = "Medium"
        if disease in ['Heart attack', 'Paralysis (brain hemorrhage)', 'AIDS', 'Tuberculosis', 'Pneumonia']:
            severity = "High"
        elif disease in ['Common Cold', 'Acne', 'Fungal infection']:
            severity = "Low"
            
        knowledge_base_list.append({
            'disease': disease,
            'description': desc,
            'precautions': prec,
            'specialist': specialist,
            'severity_level': severity
        })
        
    init_db()
    populate_knowledge_base(knowledge_base_list)
    print("Database Knowledge Base populated with disease details.")

def generate_text_dataset():
    """
    Transforms the binary structured dataset into natural-language symptom descriptions.
    This creates the text dataset needed to train the BERT-Tiny classifier and TF-IDF models.
    """
    train_path = os.path.join(RAW_DATA_DIR, 'training_data.csv')
    test_path = os.path.join(RAW_DATA_DIR, 'test_data.csv')
    
    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)
    
    # Clean column headers
    df_train.columns = [c.strip() for c in df_train.columns]
    df_test.columns = [c.strip() for c in df_test.columns]
    
    # Filter columns to only include shared symptom columns
    symptom_cols = [c for c in df_train.columns if c in df_test.columns and c != 'prognosis' and not c.startswith('Unnamed:')]

    
    # Simple templates to make text realistic
    templates = [
        "I feel like I have {symptoms}.",
        "I am suffering from {symptoms}.",
        "I have been experiencing {symptoms}.",
        "My symptoms include {symptoms}.",
        "Recently I got {symptoms}.",
        "I am showing signs of {symptoms}."
    ]
    
    def row_to_text(row, idx):
        active_symptoms = [clean_symptom_name(sym) for sym in symptom_cols if row[sym] == 1]
        if not active_symptoms:
            return "I am feeling fine, no symptoms."
        
        symptoms_str = ", ".join(active_symptoms[:-1]) + (f", and {active_symptoms[-1]}" if len(active_symptoms) > 1 else active_symptoms[0])
        template = templates[idx % len(templates)]
        return template.format(symptoms=symptoms_str)
        
    # Process Train
    train_texts = [row_to_text(row, i) for i, row in df_train.iterrows()]
    df_train_text = pd.DataFrame({
        'text': train_texts,
        'label': df_train['prognosis'].apply(clean_disease_name)
    })
    df_train_text.to_csv(os.path.join(PROCESSED_DATA_DIR, 'train_text.csv'), index=False)
    
    # Process Test
    test_texts = [row_to_text(row, i) for i, row in df_test.iterrows()]
    df_test_text = pd.DataFrame({
        'text': test_texts,
        'label': df_test['prognosis'].apply(clean_disease_name)
    })
    df_test_text.to_csv(os.path.join(PROCESSED_DATA_DIR, 'test_text.csv'), index=False)
    print("Generated natural language training and testing text datasets.")

if __name__ == '__main__':
    download_datasets()
    run_eda_and_generate_plots()
    load_and_sync_knowledge_base()
    generate_text_dataset()
