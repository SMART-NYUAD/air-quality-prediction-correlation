# Air Quality Prediction and Correlation

## Description

This repository contains a comprehensive project focused on analyzing and predicting air quality metrics, specifically PM2.5 concentrations. The project is divided into two main parts: **Air Quality Prediction** and **Air Quality Correlation**.

### Features

- **Air Quality Prediction**:
  - Training machine learning models to predict PM2.5 levels.
  - Benchmarking different models to evaluate their performance.
  - Forecasting future PM2.5 levels based on trained models.
  - Web scraping to obtain the latest PM2.5 data.

- **Air Quality Correlation**:
  - Analysis of the correlation between PM2.5 levels and other variables such as humidity, temperature, and people density.

## Repository Structure

```
air-quality-prediction-correlation/
├── air_quality_prediction/
│   ├── models/
│   ├── scripts/
│   ├── depricated/
│   ├── .ipynb_checkpoints/
│   ├── __pycache__/
│   └── outdoor_pm25_dataset/
├── air_quality_correlation/
│   ├── scripts/
├── .env
├── .gitignore
├── LICENSE
├── README.md
└── requirements.txt
```

## Getting Started

### Installation

1. **Clone the Repository**:
   ```sh
   git clone https://github.com/yourusername/air-quality-prediction-correlation.git
   cd air-quality-prediction-correlation
   ```

2. **Install Dependencies**:
   ```sh
   pip install -r requirements.txt
   ```

### Usage

- For air quality prediction tasks, navigate to the `air_quality_prediction` directory and follow the instructions in its README.
- For air quality correlation analysis, navigate to the `air_quality_correlation` directory and follow the instructions in its README.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
