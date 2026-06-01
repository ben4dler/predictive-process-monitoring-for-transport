import os
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import GroupShuffleSplit

# ── Pfade ────────────────────────────────────────────────────────────────────
BASE_DIR       = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EVENT_LOGS_DIR = os.path.join(BASE_DIR, "training_service", "data_pipeline", "event_logs")
ARTIFACTS_DIR  = os.path.join(BASE_DIR, "model_registry", "scaler")
os.makedirs(ARTIFACTS_DIR, exist_ok=True)

# ── Feature-Definitionen ─────────────────────────────────────────────────────
CATEGORICAL_COLS = ["activity", "weather_condition", "black_swan_event"]

FEATURE_COLS = [
    "activity",
    "weather_condition",
    "traffic_index",
    "weekday",
    "is_holiday",
    "delay_so_far",
    "step_index",
    "percent_done",
    "black_swan_event",
    "black_swan_delay",
]

LSTM_SEQUENCE_LENGTH = 5   


# ── Haupt-Entry-Point ────────────────────────────────────────────────────────
def preprocessing(model: str):
    data = _load_and_encode()

    X      = data[FEATURE_COLS].values
    groups = data["case_id"].values

    if model == "lstm":
        y, label_encoder = _encode_target_classification(data)
    elif model == "xgboost":
        y             = data["label_remaining_time"].values
        label_encoder = None
    else:
        raise ValueError(f"Unbekanntes Modell: '{model}'.")

    train_idx, test_idx = _case_level_split_indices(X, y, groups)

    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    scaler         = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)
    joblib.dump(scaler, os.path.join(ARTIFACTS_DIR, f"scaler_{model}.pkl"))

    if model == "lstm":
        data_train = data.iloc[train_idx].reset_index(drop=True)
        data_test  = data.iloc[test_idx].reset_index(drop=True)

        X_train_3d, y_train_seq = _build_sequences(data_train, X_train_scaled, y_train)
        X_test_3d,  y_test_seq  = _build_sequences(data_test,  X_test_scaled,  y_test)

        n_classes = len(np.unique(y))
        return (
            X_train_3d, X_test_3d,
            y_train_seq, y_test_seq,
            n_classes,
            scaler, label_encoder,
        )

    elif model == "xgboost":
        return X_train_scaled, X_test_scaled, y_train, y_test, scaler


# ── Interne Hilfsfunktionen ──────────────────────────────────────────────────
def _load_and_encode() -> pd.DataFrame:

    filepath = os.path.join(EVENT_LOGS_DIR, "transport_event_log_labeled.csv")
    data = pd.read_csv(filepath, parse_dates=["timestamp"])

    for col in CATEGORICAL_COLS:
        le = LabelEncoder()
        data[col] = le.fit_transform(data[col].astype(str))
        joblib.dump(le, os.path.join(ARTIFACTS_DIR, f"le_{col}.pkl"))

    data["is_holiday"] = data["is_holiday"].astype(int)

    data = data.drop(columns=["timestamp", "planned_eta"], errors="ignore")

    return data


def _encode_target_classification(data: pd.DataFrame):
    """Encodiert das Klassifikations-Target für das LSTM deterministisch."""
    le = LabelEncoder()
    y  = le.fit_transform(data["label_next_activity"].astype(str))
    joblib.dump(le, os.path.join(ARTIFACTS_DIR, "le_label_next_activity.pkl"))
    return y, le


def _case_level_split_indices(X, y, groups):
    """Gibt Indices zurück statt Arrays — damit data.iloc[] noch funktioniert."""
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(gss.split(X, y, groups=groups))
    return train_idx, test_idx


def _build_sequences(
    data: pd.DataFrame,       
    X_scaled: np.ndarray,
    y: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:

    X_seq, y_seq = [], []
    n_features   = X_scaled.shape[1]
    case_ids     = data["case_id"].values

    for case_id in np.unique(case_ids):
        mask     = case_ids == case_id 
        features = X_scaled[mask]
        labels   = y[mask]

        for i in range(len(features)):
            start = max(0, i - LSTM_SEQUENCE_LENGTH + 1)
            seq   = features[start : i + 1]

            if len(seq) < LSTM_SEQUENCE_LENGTH:
                pad = np.zeros((LSTM_SEQUENCE_LENGTH - len(seq), n_features))
                seq = np.vstack([pad, seq])

            X_seq.append(seq)
            y_seq.append(labels[i])

    return np.array(X_seq), np.array(y_seq)