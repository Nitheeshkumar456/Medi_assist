import sqlite3
import os
import shutil
from werkzeug.security import generate_password_hash, check_password_hash

# Determine if running in Vercel or serverless environment
IS_VERCEL = os.environ.get('VERCEL') == '1' or os.environ.get('AWS_LAMBDA_FUNCTION_NAME') is not None

ORIGINAL_DB_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'processed', 'medi_assist.db')

if IS_VERCEL:
    DB_PATH = '/tmp/medi_assist.db'
    # Copy pre-populated DB to writeable /tmp directory on startup
    if os.path.exists(ORIGINAL_DB_PATH) and not os.path.exists(DB_PATH):
        try:
            os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
            shutil.copy2(ORIGINAL_DB_PATH, DB_PATH)
            print("Successfully copied pre-populated SQLite database to /tmp.")
        except Exception as e:
            print(f"Warning: Failed to copy database to /tmp: {e}")
else:
    DB_PATH = ORIGINAL_DB_PATH

def get_db_connection():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # 1. Users table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            email TEXT,
            age INTEGER,
            gender TEXT,
            medical_history TEXT
        )
    ''')
    
    # 2. Conversations table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS conversations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            message TEXT NOT NULL,
            sender TEXT NOT NULL, -- 'user' or 'bot'
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE
        )
    ''')
    
    # 3. Knowledge base table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS knowledge_base (
            disease TEXT PRIMARY KEY,
            description TEXT NOT NULL,
            precautions TEXT NOT NULL,
            severity_level TEXT,
            specialist TEXT NOT NULL
        )
    ''')
    
    conn.commit()
    conn.close()

# --- User Authentication Functions ---

def register_user(username, password, email=None, age=None, gender=None, medical_history=None):
    conn = get_db_connection()
    cursor = conn.cursor()
    password_hash = generate_password_hash(password)
    try:
        cursor.execute('''
            INSERT INTO users (username, password_hash, email, age, gender, medical_history)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (username, password_hash, email, age, gender, medical_history))
        conn.commit()
        user_id = cursor.lastrowid
        return user_id
    except sqlite3.IntegrityError:
        return None  # Username already exists
    finally:
        conn.close()

def authenticate_user(username, password):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM users WHERE username = ?', (username,))
    user = cursor.fetchone()
    conn.close()
    
    if user and check_password_hash(user['password_hash'], password):
        return dict(user)
    return None

def update_user_profile(user_id, email, age, gender, medical_history):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('''
        UPDATE users
        SET email = ?, age = ?, gender = ?, medical_history = ?
        WHERE id = ?
    ''', (email, age, gender, medical_history, user_id))
    conn.commit()
    conn.close()

# --- Chat History Functions ---

def save_chat_message(user_id, message, sender):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO conversations (user_id, message, sender)
        VALUES (?, ?, ?)
    ''', (user_id, message, sender))
    conn.commit()
    conn.close()

def get_chat_history(user_id, limit=50):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('''
        SELECT message, sender, timestamp FROM conversations
        WHERE user_id = ?
        ORDER BY timestamp ASC
        LIMIT ?
    ''', (user_id, limit))
    history = [dict(row) for row in cursor.fetchall()]
    conn.close()
    return history

def clear_chat_history(user_id):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('DELETE FROM conversations WHERE user_id = ?', (user_id,))
    conn.commit()
    conn.close()

# --- Knowledge Base Functions ---

def populate_knowledge_base(disease_info_list):
    """
    Populates or updates the knowledge base with disease descriptions, precautions, and specialists.
    disease_info_list: list of dicts, keys: ['disease', 'description', 'precautions', 'specialist', 'severity_level']
    """
    conn = get_db_connection()
    cursor = conn.cursor()
    for info in disease_info_list:
        cursor.execute('''
            INSERT OR REPLACE INTO knowledge_base (disease, description, precautions, severity_level, specialist)
            VALUES (?, ?, ?, ?, ?)
        ''', (info['disease'], info['description'], info['precautions'], info.get('severity_level', 'Medium'), info['specialist']))
    conn.commit()
    conn.close()

def get_disease_info(disease):
    conn = get_db_connection()
    cursor = conn.cursor()
    # Perform case-insensitive search
    cursor.execute('SELECT * FROM knowledge_base WHERE LOWER(disease) = LOWER(?)', (disease.strip(),))
    row = cursor.fetchone()
    conn.close()
    if row:
        return dict(row)
    return None

if __name__ == '__main__':
    init_db()
    print("Database initialized successfully.")
