# EIS - Event Information System

A predictive process monitoring system for transport that uses machine learning to forecast remaining time and next activities in business processes.

## Description

EIS is a FastAPI-based application that provides real-time predictions for process monitoring. It uses LSTM neural networks for next activity prediction and XGBoost for remaining time estimation. The system automatically trains models on startup if they don't exist, then serves predictions via REST API endpoints.

## Features

- **Remaining Time Prediction** — Forecast how long a process case will take to complete
- **Next Activity Prediction** — Predict the next activity in a process using LSTM
- **Combined Predictions** — Get all predictions for a case at once
- **Health Check** — Monitor server status
- **Auto-Training** — Automatically trains models if saved models aren't found
- **FastAPI** — Modern, fast web framework with automatic API documentation

## Installation

### Requirements

- Python 3.13+
- pip (Python package manager)

### Setup

1. Clone the repository:
```bash
git clone https://git01lab.cs.univie.ac.at/adlerb88/eis.git
cd eis-main
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Navigate to the source directory:
```bash
cd src
```

## Usage

### Starting the Server

Run the application from the `src` directory:

```bash
python3 main.py
```

**First run:** The application will detect missing models, train them, save them, and then start the server.

**Subsequent runs:** The application will load existing models and start the server immediately.

The server runs at `http://localhost:8000`

### API Endpoints

#### Health Check
```bash
GET /health
```
Returns: `{"status": "healthy"}`

#### Predict Remaining Time
```bash
POST /predict/remaining-time
Content-Type: application/json

{
  "case_id": "case_123",
  "prefix": ["activity_1", "activity_2"]
}
```

#### Predict Next Activity
```bash
POST /predict/next_activity
Content-Type: application/json

{
  "case_id": "case_123",
  "prefix": ["activity_1", "activity_2"]
}
```

#### Get All Predictions
```bash
POST /predict/all
Content-Type: application/json

{
  "case_id": "case_123",
  "prefix": ["activity_1", "activity_2"]
}
```

### API Documentation

Interactive API documentation is available at:
- **Swagger UI:** `http://localhost:8000/docs`
- **ReDoc:** `http://localhost:8000/redoc`

## Project Structure

```
eis-main/
├── src/
│   ├── main.py                          # FastAPI application entry point
│   ├── inference_service/
│   │   ├── predictor.py                 # Prediction logic
│   │   └── inference_preprocessor.py    # Data preprocessing
│   ├── training_service/
│   │   ├── models/
│   │   │   ├── lstm_trainer.py          # LSTM model training
│   │   │   ├── xgboost_trainer.py       # XGBoost model training
│   │   │   ├── lstm_model.h5            # Saved LSTM model
│   │   │   └── xgboost_model.pkl        # Saved XGBoost model
│   │   └── data_pipeline/
│   ├── utils/
│   │   └── schemas.py                   # Request/response schemas
│   └── requirements.txt                 # Python dependencies
└── README.md                            # This file
```

## Models

- **LSTM** — Long Short-Term Memory neural network for sequence prediction (next activity)
- **XGBoost** — Gradient boosting model for remaining time prediction

## Troubleshooting

### ImportError: No module named 'src'
Run the application from the `src` directory:
```bash
cd src
python3 main.py
```

### ModuleNotFoundError: Missing dependencies
Install all requirements:
```bash
pip install -r requirements.txt
```

### Models not training
Ensure the `training_service/models/` directory exists and is writable.

## Support

For issues or questions, please contact the project maintainers or open an issue in the GitLab repository.

## Contributing

Contributions are welcome! Please follow these steps:

1. Create a feature branch (`git checkout -b feature/AmazingFeature`)
2. Commit your changes (`git commit -m 'Add AmazingFeature'`)
3. Push to the branch (`git push origin feature/AmazingFeature`)
4. Open a Merge Request

## Authors

- Benedikt Adler

## License

This project is part of the University of Vienna. See your repository settings for license details.

## Project Status

**Active Development** — Features and improvements are ongoing.