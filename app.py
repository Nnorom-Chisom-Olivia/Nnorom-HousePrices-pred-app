from flask import Flask, request, render_template
import joblib
import os
import pandas as pd

app = Flask(__name__)

# Load trained RandomForest model
# Recreate feature columns from training data
model = joblib.load("house_price_model.pkl")
feature_columns = joblib.load("house_columns.joblib")

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        # ... Paste all your prediction logic here ...
        return render_template("index.html", prediction=result)
    return render_template("index.html")
    try:
        # Get form inputs
        bedrooms = float(request.form['bedrooms'])
        size = float(request.form['size'])
        neighborhood = request.form['location']

        # Create empty input row with all features = 0
        input_data = pd.DataFrame([[0]*len(feature_columns)], columns=feature_columns)

        # Fill numeric features
        if 'BedroomAbvGr' in input_data.columns:
            input_data['BedroomAbvGr'] = bedrooms
        if 'GrLivArea' in input_data.columns:
            input_data['GrLivArea'] = size

        # Encode neighborhood
        col_name = f"Neighborhood_{neighborhood}"
        if col_name in input_data.columns:
            input_data[col_name] = 1

        # Predict
        prediction = model.predict(input_data)[0]
        result = f"${prediction:,.2f}"

    except Exception as e:
        result = f"Error: {str(e)}"

    return render_template("index.html", prediction=result)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=False)