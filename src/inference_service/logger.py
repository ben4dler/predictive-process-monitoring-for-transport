import sqlite3
import json
import os
from datetime import datetime

DB_PATH = os.path.join(os.path.dirname(__file__), "..", "logs", "predictions.db")

def init_db():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            case_id         TEXT    NOT NULL,
            task            TEXT    NOT NULL,
            prefix_length   INTEGER,
            prediction      TEXT,
            model_used      TEXT,
            request_time    TEXT,
            response_time   TEXT
        )
    """)
    conn.commit()
    conn.close()

def log_prediction(case_id, task, prefix_length, prediction, model_used):
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        INSERT INTO predictions 
        (case_id, task, prefix_length, prediction, model_used, request_time)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (
        case_id,
        task,
        prefix_length,
        json.dumps(prediction),
        model_used,
        datetime.now().isoformat()
    ))
    conn.commit()
    conn.close()

def get_history(case_id: str = None):
    conn   = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    if case_id:
        cursor.execute(
            "SELECT * FROM predictions WHERE case_id = ?", 
            (case_id,)
        )
    else:
        cursor.execute("SELECT * FROM predictions")
    
    rows = cursor.fetchall()
    conn.close()
    return rows