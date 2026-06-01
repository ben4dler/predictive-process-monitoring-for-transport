from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint
import numpy as np
import matplotlib.pyplot as plt
import os
from training_service.models.performance_printer import print_lstm_model 
from utils.preprocessor import preprocessing
from keras import Input

def create_model(n_timesteps: int, n_features: int, n_classes: int):
    model = Sequential([
        Input(shape=(n_timesteps, n_features)),  
        LSTM(64, activation='tanh', return_sequences=False),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dense(n_classes, activation='softmax')
    ])
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

def lstm(load_from_memory: bool = False, output: bool = True):

    (X_train, X_test, y_train, y_test, n_classes, scaler, label_encoder) = preprocessing(model='lstm')
    _, n_timesteps, n_features = X_train.shape


    model = create_model(n_timesteps, n_features, n_classes)


    CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
    BASE_DIR = os.path.dirname(os.path.dirname(CURRENT_DIR))
    TRAINING_DIR = os.path.join(BASE_DIR, "model_registry", "models", "lstm")
    checkpoint_path = os.path.join(TRAINING_DIR, "lstm_next_activity.keras")
    
    if not os.path.exists(TRAINING_DIR):
        os.makedirs(TRAINING_DIR)

    if load_from_memory:
        # FALL 1: Nur Laden und Evaluieren
        print(f"Lade bestehende Gewichte aus: {checkpoint_path}")
        if os.path.exists(checkpoint_path):
            load_model(checkpoint_path)
            loss, accuracy = model.evaluate(X_test, y_test)
            print(f"[LSTM] Loaded Test Loss:     {(loss):.4f}")            
            print(f"[LSTM] Loaded Test Accuracy: {(accuracy*100):.4f}%")
            
        else:
            print("FEHLER: Checkpoint-Datei nicht gefunden!")
    
    
    else: 
        # FALL 2: Neu trainieren
        print("Starte neues Training...")
        cp_callback = ModelCheckpoint(filepath=checkpoint_path,
                                     save_weights_only=False,
                                     verbose=1,
                                     save_best_only=True)

        history = model.fit(
            X_train, y_train,
            epochs          = 20,
            validation_split= 0.32,
            verbose         = 2,
            callbacks       = [cp_callback],
        )

        loss, accuracy = model.evaluate(X_test, y_test)    
        print(f"[LSTM] New Test Loss:     {loss:.4f}")
        print(f"[LSTM] New Test Accuracy: {accuracy*100:.4f}")
        
        if output: 
            print_lstm_model(history=history)
