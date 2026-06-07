import re
import os
import pandas as pd
from src.data_processor import preprocess_text, clean_symptom_name

# Standard list of symptoms in the Kaggle dataset
# We can load these dynamically from the training columns
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
TRAIN_DATA_PATH = os.path.join(BASE_DIR, 'data', 'raw', 'training_data.csv')

# Predefined common synonyms to map natural user phrases to dataset symptoms
SYNONYM_MAP = {
    'stomach ache': 'stomach pain',
    'tummy ache': 'stomach pain',
    'abdominal ache': 'abdominal pain',
    'belly pain': 'abdominal pain',
    'temperature': 'high fever',
    'hot body': 'high fever',
    'puke': 'vomiting',
    'puking': 'vomiting',
    'throw up': 'vomiting',
    'throwing up': 'vomiting',
    'headache': 'headache',
    'migraine': 'headache',
    'dizzy': 'mild fever',
    'shivering': 'shivering',
    'chills': 'chills',
    'cold': 'continuous sneezing',
    'sneezing': 'continuous sneezing',
    'cough': 'cough',
    'coughing': 'cough',
    'fatigue': 'fatigue',
    'tired': 'fatigue',
    'tiredness': 'fatigue',
    'weak': 'muscle weakness',
    'weakness': 'muscle weakness',
    'lethargic': 'lethargy',
    'lethargy': 'lethargy',
    'sore throat': 'throat irritation',
    'throat pain': 'throat irritation',
    'itching': 'itching',
    'itchy': 'itching',
    'skin rash': 'skin rash',
    'rash': 'skin rash',
    'red spots': 'red spots over body',
    'spots': 'red spots over body',
    'joint pain': 'joint pain',
    'joints hurt': 'joint pain',
    'muscle pain': 'muscle pain',
    'chest pain': 'chest pain',
    'breathless': 'breathlessness',
    'shortness of breath': 'breathlessness',
    'difficulty breathing': 'breathlessness',
    'constipation': 'constipation',
    'diarrhoea': 'diarrhoea',
    'diarrhea': 'diarrhoea',
    'loose motion': 'diarrhoea',
    'yellow skin': 'yellowish skin',
    'yellow eyes': 'yellowing of eyes',
    'loss of appetite': 'loss of appetite',
    'not hungry': 'loss of appetite',
    'nausea': 'nausea',
    'nauseous': 'nausea'
}

# Critical emergency symptoms that require urgent medical attention
EMERGENCY_KEYWORDS = [
    'chest pain', 'cannot breathe', 'difficulty breathing', 'breathlessness',
    'unconscious', 'heart attack', 'stroke', 'heavy bleeding', 'severe chest pain',
    'sudden paralysis', 'sudden weakness', 'slurred speech'
]

class NLPPipeline:
    def __init__(self):
        self.symptoms_list = []
        self.symptoms_display_map = {} # Display name to column name
        self.load_symptoms()
        
    def load_symptoms(self):
        if os.path.exists(TRAIN_DATA_PATH):
            try:
                df = pd.read_csv(TRAIN_DATA_PATH)
                df.columns = [c.strip() for c in df.columns]
                cols = [c for c in df.columns if c != 'prognosis' and not c.startswith('Unnamed:')]
                self.symptoms_list = cols
                for col in cols:
                    display_name = clean_symptom_name(col)
                    self.symptoms_display_map[display_name] = col
            except Exception as e:
                print(f"Error loading symptoms from dataset in pipeline: {e}")
        else:
            # Fallback placeholder if training file is not loaded yet
            print("Warning: training_data.csv not found. Pipeline will load symptoms later.")
            
    def get_all_symptom_display_names(self):
        if not self.symptoms_list:
            self.load_symptoms()
        return list(self.symptoms_display_map.keys())

    def classify_intent(self, text):
        """
        Determines user intent from input text:
        - 'greeting': user says hello
        - 'goodbye': user says bye/thank you
        - 'info_request': user asks about a specific disease
        - 'symptom_description': user lists their symptoms (default)
        """
        text_lower = text.lower().strip()
        
        # Greetings
        if re.search(r'\b(hi|hello|hey|greetings|good morning|good afternoon|good evening|howdy)\b', text_lower):
            return 'greeting'
            
        # Goodbyes / Exit / Thanks
        if re.search(r'\b(bye|goodbye|exit|quit|thanks|thank you|see you|no thank you)\b', text_lower):
            return 'goodbye'
            
        # Info requests (e.g. "What is malaria?", "Describe typhoid", "Explain tuberculosis")
        if re.search(r'\b(what is|define|tell me about|explain|describe|information on|info about)\b', text_lower):
            return 'info_request'
            
        return 'symptom_description'

    def check_emergency(self, text):
        """Returns True if the text contains severe, life-threatening symptoms"""
        text_lower = text.lower()
        for kw in EMERGENCY_KEYWORDS:
            if kw in text_lower:
                return True
        return False

    def extract_disease_from_query(self, text):
        """Extracts the disease name from informational queries like 'What is malaria?'"""
        text_lower = text.lower().strip()
        # Remove common question prefixes
        cleaned = re.sub(r'^(what is|define|tell me about|explain|describe|information on|info about|show details of)\s+', '', text_lower)
        cleaned = cleaned.replace('?', '').strip()
        return cleaned

    def extract_symptoms(self, text):
        """
        Extracts symptom matches from natural language.
        Returns:
            list of matching column names from the dataset.
        """
        if not self.symptoms_list:
            self.load_symptoms()
            
        text_cleaned = preprocess_text(text)
        text_lower = text.lower()
        
        matched_cols = set()
        
        # 1. Check direct synonyms mapping (e.g., if user writes 'stomach ache', we map to 'stomach pain')
        for synonym, standard_display in SYNONYM_MAP.items():
            if synonym in text_lower:
                col = self.symptoms_display_map.get(standard_display)
                if col:
                    matched_cols.add(col)
                    
        # 2. Substring matching using display names
        # E.g. if the standard display name is 'headache' and user mentions 'headache'
        for display_name, col in self.symptoms_display_map.items():
            # Match word borders to avoid partial matches (like 'cough' matching 'coughing' is fine, but avoid wrong matches)
            # We check if the display name is in the cleaned/preprocessed text
            # OR if the display name is present in the lowercased text
            if display_name in text_lower or preprocess_text(display_name) in text_cleaned:
                matched_cols.add(col)
                
        # 3. Handle specific multi-word symptom matches
        # e.g., if someone says "pain in my joints", we extract "joint pain"
        if "pain in" in text_lower or "hurts" in text_lower:
            if "joint" in text_lower:
                matched_cols.add(self.symptoms_display_map.get("joint pain"))
            if "muscle" in text_lower:
                matched_cols.add(self.symptoms_display_map.get("muscle pain"))
            if "chest" in text_lower:
                matched_cols.add(self.symptoms_display_map.get("chest pain"))
            if "stomach" in text_lower or "abdomen" in text_lower:
                matched_cols.add(self.symptoms_display_map.get("stomach pain"))
                
        return list(matched_cols)

    def text_to_binary_vector(self, text):
        """Converts user text input to a 132-dimensional binary features vector matching the dataset"""
        if not self.symptoms_list:
            self.load_symptoms()
            
        extracted = self.extract_symptoms(text)
        vector = [0] * len(self.symptoms_list)
        
        for symptom in extracted:
            if symptom in self.symptoms_list:
                idx = self.symptoms_list.index(symptom)
                vector[idx] = 1
                
        return vector, extracted

if __name__ == '__main__':
    pipeline = NLPPipeline()
    test_text = "I have a headache, stomach ache, and severe chest pain."
    intent = pipeline.classify_intent(test_text)
    vector, extracted = pipeline.text_to_binary_vector(test_text)
    is_emergency = pipeline.check_emergency(test_text)
    
    print(f"Intent: {intent}")
    print(f"Is Emergency? {is_emergency}")
    print(f"Extracted: {extracted}")
    print(f"Vector sum: {sum(vector)}")
