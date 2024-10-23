from flask import Flask, render_template, request
import pandas as pd
import joblib

# Initialize the Flask app
app = Flask(__name__)

# Load the trained model
model = joblib.load('random_forest_model.pkl')

# List of encoded columns used in the trained model
encoded_columns = model.feature_names_in_

# Function to preprocess the input data
def preprocess_input(crop, season, state, area, production, annual_rainfall, fertilizer, pesticide):
    # Create a dictionary from the input data
    input_data = {
        'Crop': [crop],
        'Season': [season],
        'State': [state],
        'Area': [area],
        'Production': [production],
        'Annual_Rainfall': [annual_rainfall],
        'Fertilizer': [fertilizer],
        'Pesticide': [pesticide]
    }

    # Convert input data to DataFrame
    input_df = pd.DataFrame(input_data)

    # Apply the same pd.get_dummies() transformation as in the original dataset
    input_df_encoded = pd.get_dummies(input_df, columns=['Crop', 'Season', 'State'])

    # Add any missing columns with a value of 0 (in case the input doesn't have all categories)
    for col in encoded_columns:
        if col not in input_df_encoded.columns:
            input_df_encoded[col] = 0

    # Reorder columns to match the training data columns
    input_df_encoded = input_df_encoded[encoded_columns]

    return input_df_encoded

# Route for the home page
@app.route('/')
def home():
    return render_template('templates/index.html')

# Route for predicting the yield
@app.route('/predict', methods=['POST'])
def predict():
    # Get the form data from the HTML form
    crop = request.form['crop']
    season = request.form['season']
    state = request.form['state']
    area = float(request.form['area'])
    production = float(request.form['production'])
    annual_rainfall = float(request.form['annual_rainfall'])
    fertilizer = float(request.form['fertilizer'])
    pesticide = float(request.form['pesticide'])

    # Preprocess input data
    input_data_encoded = preprocess_input(crop, season, state, area, production, annual_rainfall, fertilizer, pesticide)

    # Predict the yield using the pre-trained model
    predicted_yield = model.predict(input_data_encoded)

    # Render the result in the HTML template
    return render_template('templates/result.html', prediction=predicted_yield[0])

if __name__ == '__main__':
    app.run(debug=True)
