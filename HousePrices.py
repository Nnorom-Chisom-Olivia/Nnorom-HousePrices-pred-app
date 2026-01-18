#import the needed libraries
import pandas as pd 
import numpy as np 
import seaborn as sns  
import joblib
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (mean_absolute_error, mean_squared_error, r2_score)

#Load the Dataset
#Load the dataset (either using the URL or local file path)
data = pd.read_csv('train.csv')

#Preview the first few rows of the dataset
print(data.head())
print(data.info())

# Drop columns that are not useful for prediction
columns_to_drop = ['Id', 'Alley', 'PoolQC', 'Fence', 'MiscFeature']
data = data.drop(columns=columns_to_drop, axis=1)

# Convert categorical variables into dummy variables 
data = pd.get_dummies(data, drop_first=True)

# Fill missing numerical values with the median of the respective columns
data = data.fillna(data.median())

# Split the data into features (X) and target variable 

X = data.drop('SalePrice', axis=1)
y = data['SalePrice']

#Step 5: Split the Dataset into Training and Testing Sets
#Use an 80-20 train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Train the Linear Regression Model
# Initialize the Linear Regression model
model = LinearRegression()

# Fit the model to the training data
model.fit(X_train, y_train)

#Make Predictions on the test set
y_pred = model.predict(X_test)

#Evaluate the Model's Performance
# Calculate evaluation metrics
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Print the evaluation results
print(f'Mean Absolute Error (MAE): {mae}')
print(f'Mean Squared Error (MSE): {mse}')
print(f'R-squared (R²): {r2}')

#VISUALIZE THE RESULTS
# Plot actual vs predicted house prices
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.6, color='blue')
plt.plot([min(y_test), max(y_test)], [min(y_pred), max(y_pred)], color='red', linewidth=2)
plt.title('Actual vs Predicted House Prices')
plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices')
plt.savefig('house_price_plot.png')
plt.show()

#Initialize the Random Forest Regressor
model = RandomForestRegressor(random_state=42)

#Fit the model to the training data
model.fit(X_train, y_train)

#Make Predictions on the test set
y_pred = model.predict(X_test)

#Evaluate the Model's Performance
# Calculate evaluation metrics
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Print the evaluation results
print(f'Mean Absolute Error (MAE): {mae}')
print(f'Mean Squared Error (MSE): {mse}')
print(f'R-squared (R²): {r2}')

#VISUALIZE THE RESULTS
# Plot actual vs predicted house prices
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.6, color='blue')
plt.plot([min(y_test), max(y_test)], [min(y_pred), max(y_pred)], color='red', linewidth=2)
plt.title('Actual vs Predicted House Prices')
plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices')
plt.savefig('random_forest_plot.png')
plt.show()


joblib.dump(X.columns.tolist(), "house_columns.joblib")

#save the model
joblib.dump(model, "house_price_model.pkl")



"""
why we needed to use the RandomForestRegressor; 
The Linear Regression model achieved a moderate performance, 
with high MAE and MSE values and an R² score of about 0.64(64%).
This indicates that the model captures some general trends 
but struggles with accuracy because it assumes a simple linear relationship between features and house prices.

In contrast, the Random Forest Regressor produced significantly lower MAE and MSE values 
and achieved an R² score of approximately 0.89(89%). 
This shows that the model is far more accurate and better at capturing complex, 
non‑linear relationships within the dataset. 
Overall, the Random Forest model provides a much stronger and 
more reliable prediction of house prices compared to Linear Regression.
"""
