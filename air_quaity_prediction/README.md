# Air Quality Prediction

## Description

This folder contains scripts and models for predicting PM2.5 levels using Recurrent Neural Networks (RNNs) and Long Short-Term Memory (LSTM) models, which are well-suited for time series forecasting. The scripts include tools for training these models, benchmarking their performance, and forecasting future PM2.5 concentrations.

## Structure

```
air_quality_prediction/
├── data/
├── models/
│   ├── model_LSTM_50epochs/
│   ├── model_LSTM_80epochs/
├── scripts/
│   ├── benchmark_pm.py
│   ├── forecast_db.py
│   ├── train_pm.py
│   ├── web_scraper_cities.py
├── depricated/
├── .ipynb_checkpoints/
└── __pycache__/
```

## Getting Started

### Installation

1. **Navigate to the Directory**:
   ```sh
   cd air_quality_prediction
   ```

2. **Install Dependencies**:
   ```sh
   pip install -r ../requirements.txt
   ```

### Usage

1. **Train a Model**:
   ```sh
   python scripts/train_pm.py
   ```

2. **Benchmark Models**:
   ```sh
   python scripts/benchmark_pm.py
   ```

3. **Forecast PM2.5 Levels**:
   ```sh
   python scripts/forecast_db.py
   ```

### Web Scraper

The `web_scraper_cities.py` script is used to scrape the latest PM2.5 data.

### Models

Trained models are stored in the `models/` directory, organized by the number of epochs and model type.

## Deprecated Scripts

The `depricated/` folder contains older versions of scripts and models that are no longer in use.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
