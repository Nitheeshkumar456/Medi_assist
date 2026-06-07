import os
import sys
import logging
import re
from flask import Flask, request, jsonify, render_template, session, redirect, url_for
import joblib
import numpy as np
import pandas as pd

# Optional imports for PyTorch and Transformers to support lightweight serverless environments
try:
    import torch
    from transformers import BertTokenizer, BertForSequenceClassification
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False

# Local imports
from src.database import (
    init_db, register_user, authenticate_user, update_user_profile,
    save_chat_message, get_chat_history, get_disease_info, clear_chat_history
)
from src.nlp_pipeline import NLPPipeline

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("MediAssistApp")

app = Flask(__name__, 
            static_folder=os.path.join(os.path.dirname(os.path.dirname(__file__)), 'static'),
            template_folder=os.path.join(os.path.dirname(os.path.dirname(__file__)), 'templates'))

app.secret_key = "mediassist_super_secret_session_key"

# Path configs
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
RAW_DATA_DIR = os.path.join(BASE_DIR, 'data', 'raw')
MODELS_DIR = os.path.join(BASE_DIR, 'models')
TRANSFORMER_PATH = os.path.join(MODELS_DIR, 'transformer_checkpoint')
LE_PATH = os.path.join(MODELS_DIR, 'label_encoder.joblib')
TRAD_MODELS_PATH = os.path.join(MODELS_DIR, 'traditional_models.joblib')

# Load models and pipeline globally
pipeline = NLPPipeline()
label_encoder = None
traditional_models = {}
transformer_model = None
transformer_tokenizer = None
if HAS_TRANSFORMERS:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
else:
    device = None

def load_inference_models():
    global label_encoder, traditional_models, transformer_model, transformer_tokenizer
    
    # 1. Load Label Encoder
    if os.path.exists(LE_PATH):
        try:
            label_encoder = joblib.load(LE_PATH)
            logger.info("Loaded Label Encoder successfully.")
        except Exception as e:
            logger.error(f"Error loading label encoder: {e}")
            
    # 2. Load Traditional Models
    if os.path.exists(TRAD_MODELS_PATH):
        try:
            traditional_models = joblib.load(TRAD_MODELS_PATH)
            logger.info(f"Loaded traditional models: {list(traditional_models.keys())}")
        except Exception as e:
            logger.error(f"Error loading traditional models: {e}")
            
    # 3. Load Transformer Model & Tokenizer
    if HAS_TRANSFORMERS and os.path.exists(TRANSFORMER_PATH):
        try:
            transformer_tokenizer = BertTokenizer.from_pretrained(TRANSFORMER_PATH)
            transformer_model = BertForSequenceClassification.from_pretrained(TRANSFORMER_PATH)
            transformer_model.to(device)
            transformer_model.eval()
            logger.info("Loaded Transformer Model successfully.")
        except Exception as e:
            logger.error(f"Error loading transformer model: {e}")

# Helper: Perform traditional inference
def predict_traditional(text, model_name='Random Forest'):
    if not label_encoder or not traditional_models:
        return None, "Models are not fully trained or loaded."
        
    model = traditional_models.get(model_name)
    if not model:
        return None, f"Model '{model_name}' not found. Select from {list(traditional_models.keys())}"
        
    vector, extracted = pipeline.text_to_binary_vector(text)
    
    if sum(vector) == 0:
        return None, "No medical symptoms could be identified from your description. Could you clarify your symptoms?"
        
    features = np.array([vector])
    pred_idx = model.predict(features)[0]
    disease = label_encoder.inverse_transform([pred_idx])[0]
    
    # Calculate probability/confidence if supported
    confidence = 1.0
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(features)[0]
        confidence = float(probs[pred_idx])
        
    return {
        'disease': disease,
        'confidence': confidence,
        'extracted_symptoms': extracted,
        'symptom_vector': vector
    }, None

# Helper: Perform transformer inference
def predict_transformer(text):
    if not label_encoder or not transformer_model or not transformer_tokenizer:
        return None, "Transformer models are not fully trained or loaded."
        
    # Check if there are symptoms extracted first, so we don't return random classifications for hello/goodbye
    _, extracted = pipeline.text_to_binary_vector(text)
    if not extracted:
         return None, "No medical symptoms could be identified from your description. Could you clarify your symptoms?"
         
    inputs = transformer_tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=64)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = transformer_model(**inputs)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=-1)
        pred_idx = torch.argmax(probs, dim=-1).item()
        confidence = probs[0][pred_idx].item()
        
    disease = label_encoder.inverse_transform([pred_idx])[0]
    
    return {
        'disease': disease,
        'confidence': confidence,
        'extracted_symptoms': extracted
    }, None

# --- WEB FRONTEND ROUTES ---

@app.route('/')
def home():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    return render_template('index.html', user=session.get('user_username'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    if 'user_id' in session:
        return redirect(url_for('home'))
        
    error = None
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        
        user = authenticate_user(username, password)
        if user:
            session['user_id'] = user['id']
            session['user_username'] = user['username']
            session['user_age'] = user['age']
            session['user_gender'] = user['gender']
            session['user_history'] = user['medical_history']
            return redirect(url_for('home'))
        else:
            error = "Invalid username or password."
            
    return render_template('login.html', error=error)

@app.route('/register', methods=['GET', 'POST'])
def register():
    if 'user_id' in session:
        return redirect(url_for('home'))
        
    error = None
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        email = request.form.get('email')
        age = request.form.get('age')
        gender = request.form.get('gender')
        medical_history = request.form.get('medical_history', '')
        
        if not username or not password:
            error = "Username and password are required."
        else:
            user_id = register_user(username, password, email, age, gender, medical_history)
            if user_id:
                # Log them in automatically
                session['user_id'] = user_id
                session['user_username'] = username
                session['user_age'] = age
                session['user_gender'] = gender
                session['user_history'] = medical_history
                return redirect(url_for('home'))
            else:
                error = "Username already exists."
                
    return render_template('register.html', error=error)

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))

@app.route('/profile', methods=['POST'])
def update_profile():
    if 'user_id' not in session:
        return jsonify({'error': 'Unauthorized'}), 401
        
    email = request.form.get('email')
    age = request.form.get('age')
    gender = request.form.get('gender')
    medical_history = request.form.get('medical_history')
    
    update_user_profile(session['user_id'], email, age, gender, medical_history)
    
    session['user_age'] = age
    session['user_gender'] = gender
    session['user_history'] = medical_history
    
    return redirect(url_for('home'))

# --- RESTful API ENDPOINTS ---

@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'healthy',
        'models_loaded': {
            'label_encoder': label_encoder is not None,
            'traditional': len(traditional_models) > 0,
            'transformer': transformer_model is not None
        },
        'device': str(device)
    })

@app.route('/history', methods=['GET'])
def history():
    if 'user_id' not in session:
        return jsonify({'error': 'Unauthorized'}), 401
    chat_log = get_chat_history(session['user_id'])
    return jsonify({'history': chat_log})

@app.route('/clear_history', methods=['POST'])
def clear_history():
    if 'user_id' not in session:
        return jsonify({'error': 'Unauthorized'}), 401
    clear_chat_history(session['user_id'])
    return jsonify({'status': 'history cleared'})

@app.route('/symptoms', methods=['GET'])
def get_symptoms():
    return jsonify({'symptoms': pipeline.get_all_symptom_display_names()})


@app.route('/predict', methods=['POST'])
def predict():
    """
    Direct prediction REST endpoint.
    JSON Request format:
    {
        "text": "I have headache and fever",
        "model_type": "traditional" | "transformer",
        "model_name": "Random Forest" (optional for traditional)
    }
    """
    data = request.get_json() or {}
    text = data.get('text', '').strip()
    model_type = data.get('model_type', 'traditional').lower()
    model_name = data.get('model_name', 'Random Forest')
    
    if not text:
        return jsonify({'error': 'No text description provided.'}), 400
        
    # Check Emergency First
    if pipeline.check_emergency(text):
        return jsonify({
            'emergency': True,
            'warning': "🚨 EMERGENCY WARNING: Your symptoms indicate a high-risk medical condition. Please seek immediate professional emergency medical care or call 911/your local emergency number."
        })
        
    # Perform Inference
    if model_type == 'transformer':
        pred, err = predict_transformer(text)
    else:
        pred, err = predict_traditional(text, model_name)
        
    if err:
        return jsonify({'error': err}), 400
        
    # Lookup Knowledge Base
    kb_info = get_disease_info(pred['disease'])
    if not kb_info:
        kb_info = {
            'description': "No description available in the knowledge base.",
            'precautions': "Rest, hydrate, and seek medical advice.",
            'severity_level': "Medium",
            'specialist': "General Physician"
        }
        
    response = {
        'emergency': False,
        'disease': pred['disease'],
        'confidence': pred['confidence'],
        'extracted_symptoms': pred['extracted_symptoms'],
        'description': kb_info['description'],
        'precautions': kb_info['precautions'],
        'specialist': kb_info['specialist'],
        'severity_level': kb_info['severity_level'],
        'model_used': f"{model_type.capitalize()} ({model_name if model_type == 'traditional' else 'BERT-Tiny'})"
    }
    
    return jsonify(response)

@app.route('/chat', methods=['POST'])
def chat():
    """
    Stateful conversational chat.
    JSON Request format:
    {
        "message": "Hello"
    }
    """
    if 'user_id' not in session:
        return jsonify({'error': 'Unauthorized'}), 401
        
    user_id = session['user_id']
    data = request.get_json() or {}
    message = data.get('message', '').strip()
    
    if not message:
        return jsonify({'error': 'Empty message.'}), 400
        
    # Save User message to History
    save_chat_message(user_id, message, 'user')
    
    # Classify Intent
    intent = pipeline.classify_intent(message)
    logger.info(f"User Message: {message} | Intent: {intent}")
    
    # Initialize session dialog state if not existing
    if 'dialog_state' not in session:
        session['dialog_state'] = 'idle'
        session['accumulated_symptoms'] = []
        session['follow_up_symptom'] = None
        session['predicted_disease'] = None
        
    bot_response = ""
    extra_data = {}
    
    # 1. Emergency Check
    if pipeline.check_emergency(message):
        bot_response = "🚨 **EMERGENCY WARNING:** Your description mentions symptoms that could indicate a life-threatening medical emergency. Please seek immediate professional medical attention at the nearest emergency room or dial emergency services (911) immediately."
        save_chat_message(user_id, bot_response, 'bot')
        session['dialog_state'] = 'idle'
        session['accumulated_symptoms'] = []
        return jsonify({
            'response': bot_response,
            'emergency': True
        })
        
    # 2. State-Based Dialog Manager
    if session['dialog_state'] == 'awaiting_follow_up':
        user_reply = message.lower().strip()
        confirmed = False
        negated = False
        
        # Check simple yes/no
        if re.search(r'\b(yes|yeah|yep|y|i do|correct|true|indeed)\b', user_reply):
            confirmed = True
        elif re.search(r'\b(no|nay|nope|n|i don\'t|false|not really)\b', user_reply):
            negated = True
            
        if confirmed and session['follow_up_symptom']:
            session['accumulated_symptoms'].append(session['follow_up_symptom'])
            bot_response = f"Thank you for confirming. I have added '{session['follow_up_symptom'].replace('_', ' ')}' to your symptoms.\n"
        elif negated:
            bot_response = "Understood. I will not include that symptom.\n"
        else:
            bot_response = "Got it. Proceeding with your symptoms.\n"
            
        # Perform prediction on accumulated symptoms
        symptoms_text = ", ".join(session['accumulated_symptoms'])
        pred, err = predict_traditional(symptoms_text, 'Random Forest')
        
        if pred:
            kb_info = get_disease_info(pred['disease'])
            kb_desc = kb_info['description'] if kb_info else "No detailed description available."
            kb_prec = kb_info['precautions'] if kb_info else "No specific precautions."
            kb_spec = kb_info['specialist'] if kb_info else "General Physician"
            kb_sev = kb_info['severity_level'] if kb_info else "Medium"
            
            bot_response += (
                f"\nBased on our analysis, the predicted health condition is **{pred['disease']}** "
                f"(Confidence: {pred['confidence']*100:.1f}%).\n\n"
                f"**Description:** {kb_desc}\n\n"
                f"**Precautionary Actions:** {kb_prec}\n\n"
                f"**Recommended specialist:** {kb_spec}\n"
                f"**Condition Severity:** {kb_sev}"
            )
            extra_data = {
                'prediction': {
                    'disease': pred['disease'],
                    'confidence': pred['confidence'],
                    'severity_level': kb_sev,
                    'specialist': kb_spec,
                    'precautions': kb_prec,
                    'extracted_symptoms': session['accumulated_symptoms']
                }
            }
        else:
            bot_response += "\nI was unable to make a diagnosis based on these symptoms. If your symptoms worsen, please visit a doctor."
            
        # Reset State
        session['dialog_state'] = 'idle'
        session['accumulated_symptoms'] = []
        session['follow_up_symptom'] = None
        
    elif intent == 'greeting':
        bot_response = f"Hello {session.get('user_username', 'there')}! I am MediAssist, your virtual healthcare assistant. Please describe the symptoms you are experiencing so I can help analyze your condition."
        
    elif intent == 'goodbye':
        bot_response = "Thank you for using MediAssist. Stay healthy! Feel free to log back in anytime if you need health consultations."
        session['dialog_state'] = 'idle'
        session['accumulated_symptoms'] = []
        
    elif intent == 'info_request':
        disease_query = pipeline.extract_disease_from_query(message)
        kb_info = get_disease_info(disease_query)
        if kb_info:
            bot_response = (
                f"Here is some information about **{kb_info['disease']}**:\n\n"
                f"**Description:** {kb_info['description']}\n\n"
                f"**Standard Precautions:** {kb_info['precautions']}\n\n"
                f"**Recommended Specialist:** {kb_info['specialist']}\n"
                f"**Severity Category:** {kb_info['severity_level']}"
            )
        else:
            bot_response = f"I couldn't find detailed information about '{disease_query}' in my medical knowledge base. Let me know if you would like me to analyze any current symptoms you are experiencing!"
            
    else:  # intent == 'symptom_description'
        # Parse symptoms from message
        vector, extracted = pipeline.text_to_binary_vector(message)
        
        if not extracted:
            bot_response = "I couldn't identify any standard symptoms from your description. Could you specify what you feel? (e.g. 'I have a fever, cough, and stomach pain')"
        elif len(extracted) < 3:
            # We have few symptoms, let's look for a follow up.
            # Perform a draft prediction to find related symptoms
            symptoms_text = ", ".join(extracted)
            pred, _ = predict_traditional(symptoms_text, 'Random Forest')
            
            follow_up_found = False
            if pred:
                # Query related symptoms for this predicted disease
                # We can find standard columns that are 1 for this disease in the dataset
                train_path = os.path.join(RAW_DATA_DIR, 'training_data.csv')
                if os.path.exists(train_path):
                    df_train = pd.read_csv(train_path)
                    disease_rows = df_train[df_train['prognosis'].str.strip().str.lower() == pred['disease'].strip().lower()]
                    if not disease_rows.empty:
                        # Find other columns where sum > 0 and column is not already extracted
                        symptom_cols = [c for c in df_train.columns if c != 'prognosis']
                        # Find most frequent symptom of this disease that was NOT reported by the user
                        for col in symptom_cols:
                            if col not in extracted and disease_rows[col].sum() > (len(disease_rows) * 0.5):
                                # Ask follow up
                                display_name = col.replace('_', ' ')
                                bot_response = f"I've noted that you are experiencing: {', '.join([s.replace('_', ' ') for s in extracted])}.\n\nDo you also experience **{display_name}**?"
                                session['dialog_state'] = 'awaiting_follow_up'
                                session['accumulated_symptoms'] = extracted
                                session['follow_up_symptom'] = col
                                follow_up_found = True
                                break
                                
            if not follow_up_found:
                # Fallback to direct prediction
                pred, _ = predict_traditional(symptoms_text, 'Random Forest')
                if pred:
                    kb_info = get_disease_info(pred['disease'])
                    kb_desc = kb_info['description'] if kb_info else "No detailed description available."
                    kb_prec = kb_info['precautions'] if kb_info else "No specific precautions."
                    kb_spec = kb_info['specialist'] if kb_info else "General Physician"
                    kb_sev = kb_info['severity_level'] if kb_info else "Medium"
                    
                    bot_response = (
                        f"Based on your symptoms, the predicted condition is **{pred['disease']}** "
                        f"(Confidence: {pred['confidence']*100:.1f}%).\n\n"
                        f"**Description:** {kb_desc}\n\n"
                        f"**Precautions:** {kb_prec}\n\n"
                        f"**Recommended Specialist:** {kb_spec}\n"
                        f"**Severity Level:** {kb_sev}"
                    )
                    extra_data = {
                        'prediction': {
                            'disease': pred['disease'],
                            'confidence': pred['confidence'],
                            'severity_level': kb_sev,
                            'specialist': kb_spec,
                            'precautions': kb_prec,
                            'extracted_symptoms': extracted
                        }
                    }
                else:
                    bot_response = "I was unable to determine a disease from your description. Could you provide a bit more detail about your symptoms?"
        else:
            # Direct prediction on 3+ symptoms
            symptoms_text = ", ".join(extracted)
            pred, _ = predict_traditional(symptoms_text, 'Random Forest')
            if pred:
                kb_info = get_disease_info(pred['disease'])
                kb_desc = kb_info['description'] if kb_info else "No detailed description available."
                kb_prec = kb_info['precautions'] if kb_info else "No specific precautions."
                kb_spec = kb_info['specialist'] if kb_info else "General Physician"
                kb_sev = kb_info['severity_level'] if kb_info else "Medium"
                
                bot_response = (
                    f"Based on your symptoms, the predicted condition is **{pred['disease']}** "
                    f"(Confidence: {pred['confidence']*100:.1f}%).\n\n"
                    f"**Description:** {kb_desc}\n\n"
                    f"**Precautions:** {kb_prec}\n\n"
                    f"**Recommended Specialist:** {kb_spec}\n"
                    f"**Severity Level:** {kb_sev}"
                )
                extra_data = {
                    'prediction': {
                        'disease': pred['disease'],
                        'confidence': pred['confidence'],
                        'severity_level': kb_sev,
                        'specialist': kb_spec,
                        'precautions': kb_prec,
                        'extracted_symptoms': extracted
                    }
                }
            else:
                bot_response = "I was unable to identify a matching condition for those symptoms. Please consult a health professional."
                
    # Save Bot Response to DB
    save_chat_message(user_id, bot_response, 'bot')
    
    return jsonify({
        'response': bot_response,
        'emergency': False,
        **extra_data
    })

if __name__ == '__main__':
    # Initialize DB tables
    init_db()
    
    # Load model binaries if they exist
    load_inference_models()
    
    # Start flask app
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
else:
    init_db()
    load_inference_models()
