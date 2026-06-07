# MediAssist: AI-Powered Medical Chatbot & Diagnostics Dashboard

MediAssist is an end-to-end medical virtual assistant and diagnostic dashboard. It is designed to accept user symptom descriptions in natural language, automatically identify medical symptoms and conversational intents, classify likely health conditions using both traditional Machine Learning and Transformer models, and log conversational histories and profiles securely.

---

## ⚕️ Clinical Disclaimer
**IMPORTANT:** MediAssist is built for educational, illustrative, and research purposes. It is **NOT** a certified medical diagnostic tool and does not provide clinical treatments or professional medical advice. If you or someone you know is experiencing severe symptoms or a life-threatening medical emergency (such as severe chest pain, shortness of breath, sudden weakness, or unconsciousness), please contact your local emergency services (911) immediately.

---

## ⚙️ Tech Stack
- **Backend:** Python 3.10+, Flask, SQLite
- **Machine Learning & NLP:** PyTorch, Scikit-Learn, Hugging Face Transformers (`prajjwal1/bert-tiny`), NLTK
- **Frontend:** HTML5, CSS3 (Premium Glassmorphic Styling), Vanilla JS (Speech-to-Text & Client translations)
- **Deployment:** Docker, Docker Compose, Gunicorn WSGI

---

## 📊 System Architecture

```mermaid
flowchart TD
    subgraph Client ["Client Interface (Premium Web UI)"]
        UI["HTML / CSS (Glassmorphism)"]
        JS["JS (AJAX, Speech-to-Text, Translations)"]
    end

    subgraph Backend ["Flask RESTful API Backend"]
        App["src/app.py (Flask Web App)"]
        DB["src/database.py (SQLite SQLite3)"]
        NLPPipe["src/nlp_pipeline.py (Intent & Symptom Extractor)"]
    end

    subgraph ML ["Machine Learning Pipelines"]
        ML_Models["src/ml_models.py (RF, LR, NB, SVM Classifier)"]
        Transformer["src/transformer_model.py (BERT-Tiny Classifier)"]
    end

    subgraph Storage ["Persistent Storage"]
        SQLiteDB["data/processed/medi_assist.db"]
        ModelCheck["models/traditional_models.joblib"]
        BERTCheck["models/transformer_checkpoint/"]
    end

    UI -->|1. Natural Language Input| JS
    JS -->|2. HTTP POST Request| App
    App -->|3. Route Message| NLPPipe
    NLPPipe -->|4. Parse Symptoms & Check Emergency| App
    App -->|5. Predict (Binary Vector / Text)| ML_Models & Transformer
    ML_Models -->|Read weights| ModelCheck
    Transformer -->|Read checkpoint| BERTCheck
    App -->|6. Retrieve description & precautions| DB
    DB -->|SQL Queries| SQLiteDB
    App -->|7. JSON Response| JS
    JS -->|8. Render Dashboard Updates| UI
```

---

## 📂 Project Structure
```
/
├── data/
│   ├── raw/                        # Original CSV datasets downloaded
│   └── processed/
│       └── medi_assist.db          # Persisted SQLite Database
├── src/
│   ├── data_processor.py           # EDA, text preprocessing, dataset download
│   ├── database.py                 # SQLite CRUD, Auth & Chat history logs
│   ├── nlp_pipeline.py             # Entity extraction & intent classification
│   ├── ml_models.py                # Traditional ML model training & evaluation
│   ├── transformer_model.py        # Transformer BERT-Tiny training & evaluation
│   └── app.py                      # Flask RESTful Server
├── static/
│   ├── css/
│   │   └── style.css               # Premium Glassmorphism styling and keyframe animations
│   └── js/
│       └── main.js                 # AJAX chat, direct predictions, Web Speech API integration
├── templates/
│   ├── index.html                  # Diagnostic Center & Checklist Dashboard
│   ├── login.html                  # User login page
│   └── register.html               # User profile signup page
├── artifacts/
│   ├── plots/                      # Generated visual metrics and EDA
│   │   ├── disease_distribution.png
│   │   ├── symptom_frequency.png
│   │   └── model_comparison.png
│   └── model_evaluation_report.md  # Detailed performance metrics comparison
├── Dockerfile                      # Application containerizer
├── docker-compose.yml              # Local compose orchestrator
├── requirements.txt                # System Python dependencies
└── README.md                       # Documentation
```

---

## 🛠️ Setup & Installation

### Local Setup
1. **Clone the project** to your local workspace directory: `c:\Final_year_projects\Medi_assist`
2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
3. **Download data & perform EDA**:
   Runs the data engine to download datasets, generate visualization distributions, and load the SQLite knowledge base.
   ```bash
   python src/data_processor.py
   ```
4. **Train and compare models**:
   Trains Logistic Regression, Random Forest, Naive Bayes, SVM, and the BERT-Tiny sequence classification transformer:
   ```bash
   python src/ml_models.py
   python src/transformer_model.py
   ```
5. **Start the Flask Backend server**:
   ```bash
   python src/app.py
   ```
6. **Access the Web Interface**: Open `http://localhost:5000` in your web browser.

---

## 🐳 Docker Deployment

To build and run the entire application in a self-contained container environment, execute:

```bash
# Start container
docker-compose up --build
```
The Docker image is optimized to pre-download the required BERT models and NLTK resources during the build step, ensuring Gunicorn runs immediately without cold-start network blocks.

---

## 🔌 API Documentation

### `POST /predict`
Performs classification based on symptom lists.
- **Request Body (JSON):**
  ```json
  {
    "text": "headache, nausea, shivering",
    "model_type": "traditional",
    "model_name": "Random Forest"
  }
  ```
- **Response (JSON):**
  ```json
  {
    "emergency": false,
    "disease": "Malaria",
    "confidence": 0.95,
    "extracted_symptoms": ["headache", "nausea", "shivering"],
    "description": "An infectious disease caused by plasmodium parasites...",
    "precautions": "Avoid mosquito bites, use bed nets, seek medical checkup",
    "specialist": "Infectious Disease Specialist",
    "severity_level": "Medium",
    "model_used": "Traditional (Random Forest)"
  }
  ```

### `POST /chat`
Conversational chat interface with emergency flagging and stateful follow-up capabilities.
- **Request Body (JSON):**
  ```json
  { "message": "I feel dizzy and have stomach ache." }
  ```
- **Response (JSON):**
  ```json
  {
    "response": "I've noted that you are experiencing dizziness and stomach pain. Do you also experience abdominal pain?",
    "emergency": false
  }
  ```

### `GET /symptoms`
Returns a list of all displayable symptoms from the database.

### `GET /history`
Returns chat logs for the authenticated session.

### `GET /health`
Returns the status of server and model loading.
