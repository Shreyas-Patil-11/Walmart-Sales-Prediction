from flask import Flask, render_template, request
import pandas as pd
import pickle

application = Flask(__name__)
app = application

# Load the scaler and trained model
with open('scaler.pkl', 'rb') as scaler_file:
    loaded_scaler = pickle.load(scaler_file)

with open('model.pkl', 'rb') as model_file:
    loaded_model = pickle.load(model_file)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input data from form
        store = int(request.form['store'])
        holiday_flag = int(request.form['holiday_flag'])
        temperature = float(request.form['temperature'])
        fuel_price = float(request.form['fuel_price'])
        cpi = float(request.form['cpi'])
        unemployment = float(request.form['unemployment'])
        day = int(request.form['day'])
        month = int(request.form['month'])
        year = int(request.form['year'])
        weekday = int(request.form['weekday'])

        # Create DataFrame
        input_data = pd.DataFrame([{
            'Store': store,
            'Holiday_Flag': holiday_flag,
            'Temperature': temperature,
            'Fuel_Price': fuel_price,
            'CPI': cpi,
            'Unemployment': unemployment,
            'day': day,
            'month': month,
            'year': year,
            'weekday': weekday
        }])

        # Standardize
        scaled_input = loaded_scaler.transform(input_data)

        # Predict
        prediction = loaded_model.predict(scaled_input)[0]

        return render_template('result.html', predicted_sales=round(prediction, 2))

    except Exception as e:
        return f"❌ Error: {e}"

if __name__ == '__main__':
    app.run(debug=True)
