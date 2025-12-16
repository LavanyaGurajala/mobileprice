# Mobile Price Range Predictor

A minimal machine learning web application that predicts mobile phone price categories based on device specifications using Python, Flask, and scikit-learn.

## Overview

This project uses a Random Forest Classifier to predict mobile price ranges into four categories:
- **Low (0)**: Budget-friendly phones
- **Medium (1)**: Mid-range phones
- **High (2)**: High-end phones
- **Premium (3)**: Flagship phones

## Dataset

The project uses two CSV files:

### train.csv
Contains 20 features and 1 target column (`price_range`) used to train the model:
- `battery_power`: Total energy a battery can store (mAh)
- `blue`: Has Bluetooth (0/1)
- `clock_speed`: Speed at which microprocessor executes instructions (GHz)
- `dual_sim`: Has dual SIM support (0/1)
- `fc`: Front camera megapixels
- `four_g`: Has 4G (0/1)
- `int_memory`: Internal memory (GB)
- `m_dep`: Mobile depth (cm)
- `mobile_wt`: Weight of mobile phone (g)
- `n_cores`: Number of processor cores
- `pc`: Primary camera megapixels
- `px_height`: Pixel resolution height
- `px_width`: Pixel resolution width
- `ram`: RAM (MB)
- `sc_h`: Screen height (cm)
- `sc_w`: Screen width (cm)
- `talk_time`: Longest time battery will last during calls (hours)
- `three_g`: Has 3G (0/1)
- `touch_screen`: Has touch screen (0/1)
- `wifi`: Has WiFi (0/1)
- `price_range`: Target variable (0-3)

### test.csv
Contains same features as training data (plus an `id` column) but without the `price_range` target, used for evaluation or batch predictions.

## Input and Output

### User Inputs (8 key specifications):
- Battery Power (mAh)
- RAM (MB)
- Internal Memory (GB)
- Primary Camera (MP)
- Clock Speed (GHz)
- Number of Cores
- Pixel Height
- Pixel Width

### Output:
Predicted price category displayed as:
- **Low Cost** (0)
- **Medium Cost** (1)
- **High Cost** (2)
- **Premium Cost** (3)

## Steps to Run

### 1. Install Requirements
```bash
pip install -r requirements.txt
```

### 2. Train the Model
```bash
python model/train_model.py
```
This will:
- Load the training data from `data/train.csv`
- Train a Random Forest Classifier
- Save the trained model as `model/mobile_price_model.pkl`
- Display validation accuracy and feature importance

### 3. Run the Flask Application
```bash
python app.py
```
The web application will start at `http://localhost:5000`

### 4. Make Predictions
- Open your browser and navigate to `http://localhost:5000`
- Enter mobile specifications in the form
- Click "Predict Price Range" to see the predicted category

## Project Structure

```
mobile-price-range-predictor/
│
├── data/
│   ├── train.csv          # Training dataset with price_range
│   └── test.csv           # Test dataset without price_range
│
├── model/
│   ├── train_model.py     # Script to train and save the model
│   └── mobile_price_model.pkl  # Trained model (generated)
│
├── app.py                 # Flask web application
├── requirements.txt       # Python dependencies
└── README.md             # Project documentation
```

## Technical Details

- **Framework**: Flask (Python web framework)
- **ML Algorithm**: Random Forest Classifier
- **ML Library**: scikit-learn
- **Data Processing**: pandas, numpy
- **Model Persistence**: joblib

## Notes

- The model uses all 20 features for training but only collects 8 key features from users
- Remaining features use typical default values from the training data
- Feature order is preserved exactly as in the training dataset to ensure accurate predictions
- No external frontend frameworks or APIs are used - everything is rendered in Flask
