# 🛒 Walmart Weekly Sales Prediction

This is a Flask web application that predicts Walmart weekly sales using Linear Regression and StandardScaler (without pretrained ML models or external files like PCA or Random Forest).

## 🚀 Features
- Predicts weekly sales using:
  - Store number
  - Holiday flag
  - Temperature
  - Fuel price
  - CPI
  - Unemployment rate
  - Date (from which day, month, year, and weekday are extracted)
- Simple UI form
- Uses `LinearRegression` and `StandardScaler` directly

## 🧠 Model Info
- Model: `LinearRegression` from `scikit-learn`
- Features used: `['Store', 'Holiday_Flag', 'Temperature', 'Fuel_Price', 'CPI', 'Unemployment', 'day', 'month', 'year', 'weekday']`
- Target: `Weekly_Sales`

## 🛠️ Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/walmart-sales-predictor.git
cd walmart-sales-predictor

# Create and activate virtual environment (optional)
python -m venv venv
source venv/bin/activate   # or venv\Scripts\activate on Windows

# Install dependencies
pip install -r requirements.txt
