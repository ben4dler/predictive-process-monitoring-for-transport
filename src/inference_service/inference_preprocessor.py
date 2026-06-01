import os
import sys
import numpy as np
import pandas as pd
import joblib

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

ARTIFACTS_DIR = os.path.join(BASE_DIR, 'model_registry', 'scaler')

CATEGORICAL_COLS     = ["activity", "weather_condition", "black_swan_event"]
FEATURE_COLS         = [
    "activity", "weather_condition", "traffic_index",
    "weekday", "is_holiday", "delay_so_far",
    "step_index", "percent_done",
    "black_swan_event", "black_swan_delay"
]
LSTM_SEQUENCE_LENGTH = 5


class InferencePreprocessor:
    """
    Verwendet die gespeicherten Artifacts aus dem Training
    um einen einzelnen API-Request zu transformieren.
    """

    def __init__(self):

        self.label_encoders = {}
        for col in CATEGORICAL_COLS:
            path = os.path.join(ARTIFACTS_DIR, f"le_{col}.pkl")
            self.label_encoders[col] = joblib.load(path)

        self.scaler_lstm    = joblib.load(
            os.path.join(ARTIFACTS_DIR, "scaler_lstm.pkl")
        )
        self.scaler_xgb     = joblib.load(
            os.path.join(ARTIFACTS_DIR, "scaler_xgboost.pkl")
        )
        self.activity_enc   = joblib.load(
            os.path.join(ARTIFACTS_DIR, "le_label_next_activity.pkl")
        )

    def _prefix_to_dataframe(self, prefix: list) -> pd.DataFrame:
            """Wandelt den rohen API-Prefix in einen DataFrame um."""

            data_dicts = [
                p.model_dump() if hasattr(p, 'model_dump') else p.dict() if hasattr(p, 'dict') else p 
                for p in prefix
            ]
            
            df = pd.DataFrame(data_dicts)

            # KATEGORIALE FEATURING
            for col in CATEGORICAL_COLS:
                if col in df.columns:
                    df[col] = self.label_encoders[col].transform(
                        df[col].astype(str)
                    )
            if 'is_holiday' not in df.columns:
                df['is_holiday'] = 0
            else:
                df['is_holiday'] = df['is_holiday'].astype(int)

            # WEITERE FEATURE-BERECHNUNGEN
            if 'step_index' not in df.columns:
                df['step_index'] = range(len(df))

            if 'percent_done' not in df.columns:
                max_steps         = 5  
                df['percent_done'] = df['step_index'] / max_steps
            df = df.drop(
                columns=['timestamp', 'planned_eta', 'resource', 'case_id'],
                errors='ignore'
            )
            return df[FEATURE_COLS]

    def for_lstm(self, prefix: list) -> np.ndarray:
        """
        Gibt eine 3D Sequenz zurück für das LSTM.
        Shape: (1, LSTM_SEQUENCE_LENGTH, n_features)
        """

        df       = self._prefix_to_dataframe(prefix)
        X        = df.values
        X_scaled = self.scaler_lstm.transform(X)


        n_features = X_scaled.shape[1]
        start      = max(0, len(X_scaled) - LSTM_SEQUENCE_LENGTH)
        seq        = X_scaled[start:]

        if len(seq) < LSTM_SEQUENCE_LENGTH:
            pad = np.zeros((LSTM_SEQUENCE_LENGTH - len(seq), n_features))
            seq = np.vstack([pad, seq])

        return seq.reshape(1, LSTM_SEQUENCE_LENGTH, n_features)

    def for_xgboost(self, prefix: list) -> np.ndarray:
        """
        Gibt einen 2D Feature-Vektor zurück für XGBoost.
        Shape: (1, n_features) — nur das letzte Event
        """

        df       = self._prefix_to_dataframe(prefix)
        X        = df.values
        X_scaled = self.scaler_xgb.transform(X)


        return X_scaled[-1].reshape(1, -1)

    def decode_activity(self, index: int) -> str:
        """Wandelt einen Index zurück in den Aktivitätsnamen."""
        return self.activity_enc.classes_[index]