from fastapi import FastAPI, HTTPException
from inference_service.predictor import Predictor
# from inference_service.logger import log_prediction
from training_service.models import lstm_trainer
from training_service.models import xgboost_trainer
from utils.schemas import PredictionRequest
import uvicorn
import os
# from training_service.data_pipeline import data_loader

app = FastAPI()
predictor = Predictor()


@app.post("/predict/remaining-time")
def predict_remaining_time(request: PredictionRequest):
    result = predictor.remaining_time(request)

    # log_prediction(
    #     case_id      = request.case_id,
    #     task         = "remaining_time",
    #     prefix_length= len(request.prefix),
    #     prediction   = result,
    #     model_used   = "xgboost"
    # )

    return result

@app.post("/predict/next_activity")
def predict_next_activity(request: PredictionRequest):
    result = predictor.next_activity(request)

    # log_prediction(
    #     case_id      = request.case_id,
    #     task         = "next_activity",
    #     prefix_length= len(request.prefix),
    #     prediction   = result,
    #     model_used   = "lstm"
    # )

    return result

@app.post("/predict/all")
def predict_all(request:PredictionRequest):
    result = predictor.all_predictions(request)

    # log_prediction(
    #     case_id      = request.case_id,
    #     task         = "all",
    #     prefix_length= len(request.prefix),
    #     prediction   = result,
    #     model_used   = "all"
    # )

    return result

@app.get("/health")
def health_check():
    return {"status": "healthy"}


def models_exist():
    """Check if trained models have been saved"""
    # Adjust these paths based on where your models are saved
    lstm_model_path = "training_service/models/lstm_model.h5"
    xgboost_model_path = "training_service/models/xgboost_model.pkl"
    
    return os.path.exists(lstm_model_path) and os.path.exists(xgboost_model_path)


if __name__ == '__main__':
    if models_exist():
        print("✓ Models found. Starting server...")
    else:
        print("✗ Models not found. Training models...")
        lstm_trainer.lstm()
        xgboost_trainer.xgboost()
        print("✓ Model training complete. Starting server...")
    
    uvicorn.run(app, host="0.0.0.0", port=8000)