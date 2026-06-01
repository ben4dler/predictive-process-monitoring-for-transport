import os
import numpy as np
import joblib
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, r2_score
from utils.preprocessor import preprocessing
from training_service.models.performance_printer import print_xgboost_model

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(os.path.dirname(CURRENT_DIR))
TRAINING_DIR = os.path.join(BASE_DIR, "model_registry", "models", "xgboost")


def xgboost(output: bool = False):
    X_train, X_test, y_train, y_test, scaler = preprocessing(model='xgboost')

    model = xgb.XGBRegressor(
        n_estimators        = 500,
        max_depth           = 6,
        learning_rate       = 0.05,
        subsample           = 0.8,
        colsample_bytree    = 0.8, 
        min_child_weight    = 5,        
        early_stopping_rounds = 30,      
        objective           = "reg:squarederror",
    )

    model.fit(
        X_train, y_train,
        eval_set            = [(X_test, y_test)],
        verbose             = False,
    )

    y_pred = model.predict(X_test)

    mae    = mean_absolute_error(y_test, y_pred)
    r2     = r2_score(y_test, y_pred)
    within_15 = np.mean(np.abs(y_test - y_pred) <= 15) * 100  
    within_30 = np.mean(np.abs(y_test - y_pred) <= 30) * 100

    print(f"[XGBOOST] MAE:             {mae:.1f} Minuten")
    print(f"[XGBOOST] R²:              {r2:.4f}")
    print(f"[XGBOOST] Accuracy ±15min: {within_15:.1f}%")
    print(f"[XGBOOST] Accuracy ±30min: {within_30:.1f}%")
    print(f"[XGBOOST] Best iteration:  {model.best_iteration}")


    os.makedirs(TRAINING_DIR, exist_ok=True)
    joblib.dump(model, os.path.join(TRAINING_DIR, "xgboost_remaining_time.pkl"))

    if output:
        print_xgboost_model(model)

    return model, scaler