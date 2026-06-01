from inference_service.inference_preprocessor import InferencePreprocessor
from inference_service.model_loader import ModelLoader
from inference_service.logger import log_prediction
import numpy as np

class Predictor:

    def __init__(self):
        self.preprocessor = InferencePreprocessor()

        loader = ModelLoader()
        self.lstm_next_activity = loader.load("lstm_next_activity")
        # self.lstm_outcome       = loader.load("lstm_outcome")
        self.xgb_remaining_time = loader.load("xgboost_remaining_time")
        # self.xgb_risk_score     = loader.load("xgboost_risk_score")

    def next_activity(self, request) -> dict:

        X_lstm = self.preprocessor.for_lstm(request.prefix)

        probabilities      = self.lstm_next_activity.predict(X_lstm)[0]
        predicted_idx      = np.argmax(probabilities)
        predicted_activity = self.preprocessor.decode_activity(predicted_idx)

        return {
            "case_id":       request.case_id,
            "prediction":    predicted_activity,
            "probabilities": {
                self.preprocessor.decode_activity(i): round(float(p), 3)
                for i, p in enumerate(probabilities)
            },
            "model": "lstm"
        }

    def remaining_time(self, request) -> dict:

        X_xgb      = self.preprocessor.for_xgboost(request.prefix)
        prediction = self.xgb_remaining_time.predict(X_xgb)[0]

        return {
            "case_id":    request.case_id,
            "prediction": round(float(prediction), 1),
            "unit":       "minutes",
            "model":      "xgboost"
        }
    def outcome(self, request) -> dict:
        
        X_scaled = self._preprocess(request.prefix)
        X_lstm   = self._to_lstm_input(X_scaled)
        
        probability = float(self.lstm_outcome.predict(X_lstm)[0][0])
        prediction  = "delayed" if probability > 0.5 else "on_time"
        
        result = {
            "case_id":     request.case_id,
            "prediction":  prediction,
            "probability": round(probability, 3),
            "model":       "lstm"
        }
        
        log_prediction(request.case_id, "outcome",
                      len(request.prefix), result, "lstm")
        return result
    
    def risk_score(self, request) -> dict:
        
        X_scaled = self._preprocess(request.prefix)
        X_xgb    = self._to_xgb_input(X_scaled)
        
        score = float(self.xgb_risk_score.predict(X_xgb)[0])
        score = min(max(score, 0.0), 1.0) 
        
        result = {
            "case_id":    request.case_id,
            "prediction": round(score, 3),
            "threshold":  30,
            "unit":       "minutes",
            "model":      "xgboost"
        }
        
        log_prediction(request.case_id, "risk_score",
                      len(request.prefix), result, "xgboost")
        return result
    
    # def all_predictions(self, request) -> dict:
    #     """Alle 4 Vorhersagen auf einmal"""
        
    #     X_scaled = self._preprocess(request.prefix)
    #     X_lstm   = self._to_lstm_input(X_scaled)
    #     X_xgb    = self._to_xgb_input(X_scaled)
        
    #     rt    = self.xgb_remaining_time.predict(X_xgb)[0]
    #     na    = self.lstm_next_activity.predict(X_lstm)[0]
    #     oc    = self.lstm_outcome.predict(X_lstm)[0][0]
    #     rs    = self.xgb_risk_score.predict(X_xgb)[0]
        
    #     return {
    #         "case_id":        request.case_id,
    #         "remaining_time": round(float(rt), 1),
    #         "next_activity":  self.activity_decoder[np.argmax(na)],
    #         "outcome":        "delayed" if float(oc) > 0.5 else "on_time",
    #         "risk_score":     round(float(rs), 3)
    #     }