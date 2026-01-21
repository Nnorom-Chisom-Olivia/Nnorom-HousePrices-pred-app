from flask import Flask, request, render_template
import joblib
import os
import pandas as pd

app = Flask(__name__)

# Load trained model and feature list from the /model/ folder
model = joblib.load("model/house_price_model.pkl")
feature_columns = joblib.load("model/house_columns.joblib")

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # 1. Grab the 6 inputs from the form
        # We put them directly into a dictionary
        input_dict = {
            'OverallQual': int(request.form['OverallQual']),
            'GrLivArea': float(request.form['GrLivArea']),
            'TotalBsmtSF': float(request.form['TotalBsmtSF']),
            'GarageCars': int(request.form['GarageCars']),
            'FullBath': int(request.form['FullBath']),
            'YearBuilt': int(request.form['YearBuilt'])
        }

        # 2. Convert dictionary to a DataFrame
        input_df = pd.DataFrame([input_dict])

        # 3. Ensure the columns are in the exact same order as the model expects
        input_df = input_df[feature_columns]

        # 4. Predict
        prediction = model.predict(input_df)[0]
        result = f"${prediction:,.2f}"

    except Exception as e:
        result = f"Error: {str(e)}"

    # We use 'prediction_text' to match your index.html display logic
    return render_template("index.html", prediction_text=f"Predicted House Price: {result}")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=False)