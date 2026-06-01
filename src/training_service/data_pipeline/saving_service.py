import os
import pandas as pd

# Relativer Pfad: von training_service/ eine Ebene hoch, dann in event_logs/
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
EVENT_LOGS_DIR = os.path.join(BASE_DIR, "event_logs")

def save_event_log(df: pd.DataFrame, filename: str):
    """Speichert den Event Log in den event_logs Ordner"""
    
    # Ordner erstellen falls er nicht existiert
    os.makedirs(EVENT_LOGS_DIR, exist_ok=True)
    
    filepath = os.path.join(EVENT_LOGS_DIR, filename)
    df.to_csv(filepath, index=False)
    print(f"Event Log gespeichert: {filepath}")
    return filepath

def load_event_log(filename: str) -> pd.DataFrame:
    """Lädt einen Event Log aus dem event_logs Ordner"""
    
    filepath = os.path.join(EVENT_LOGS_DIR, filename)
    
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Event Log nicht gefunden: {filepath}")
    
    return pd.read_csv(filepath, parse_dates=["timestamp", "planned_eta"])