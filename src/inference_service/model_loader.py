import pickle
import os
from tensorflow.keras.models import load_model

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(CURRENT_DIR) 
MODELS_DIR = os.path.join(BASE_DIR, "model_registry", "models")
SCALER_DIR = os.path.join(BASE_DIR, "model_registry", "scaler")



class ModelLoader:
    
    def load(self, model_name: str):
        """Lädt LSTM (.h5) oder XGBoost (.pkl) Modell"""

        lstm_path = os.path.join(MODELS_DIR, "lstm", f"{model_name}.keras")
        xgb_path = os.path.join(MODELS_DIR, "xgboost", f"{model_name}.pkl")
        
        if os.path.exists(lstm_path):
            print(f"-> Lade LSTM aus: {lstm_path}")
            return load_model(lstm_path)
        
        elif os.path.exists(xgb_path):
            print(f"-> Lade XGBoost aus: {xgb_path}")
            with open(xgb_path, "rb") as f:
                return pickle.load(f)
        
        raise FileNotFoundError(
            f"Modell '{model_name}' wurde nirgends gefunden.\n"
            f"Gesuchte Pfade:\n"
            f" - LSTM: {lstm_path}\n"
            f" - XGB:  {xgb_path}"
        )
    
    def load_scaler(self):
        path = os.path.join(SCALER_DIR, "scaler.pkl")
        with open(path, "rb") as f:
            return pickle.load(f)
    
    def load_encoder(self, name: str):
        path = os.path.join(SCALER_DIR, f"encoder_{name}.pkl")
        with open(path, "rb") as f:
            return pickle.load(f)
    
    def load_decoder(self, name: str):
        path = os.path.join(SCALER_DIR, f"decoder_{name}.pkl")
        with open(path, "rb") as f:
            return pickle.load(f)