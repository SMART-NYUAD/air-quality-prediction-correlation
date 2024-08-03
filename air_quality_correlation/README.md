# Air Quality Correlation

## Description

This folder contains scripts for analyzing the correlation between PM2.5 levels and other variables such as humidity, temperature, and people density. These analyses help understand the factors affecting air quality.

## Structure

```
air_quality_correlation/
├── scripts/
│   ├── pm2_5_humidity.py
│   ├── pm2_5_people_count.py
│   ├── pm2_5_temperature.py
│   ├── pm_2_5_out_ind.py
│   ├── triple_correlation.py
```

## Getting Started

### Installation

1. **Navigate to the Directory**:
   ```sh
   cd air_quality_correlation
   ```

2. **Install Dependencies**:
   ```sh
   pip install -r ../requirements.txt
   ```

### Usage

1. **Analyze PM2.5 and Humidity Correlation**:
   ```sh
   python scripts/pm2_5_humidity.py
   ```

2. **Analyze PM2.5 and Temperature Correlation**:
   ```sh
   python scripts/pm2_5_temperature.py
   ```

3. **Triple Correlation Analysis**:
   ```sh
   python scripts/triple_correlation.py
   ```

4. **Analyze PM2.5 and People Density Correlation**:
   ```sh
   python scripts/pm2_5_people_count.py
   ```

5. **Analyze Indoor and Outdoor PM2.5 Correlation**:
   ```sh
   python scripts/pm_2_5_out_ind.py
   ```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
