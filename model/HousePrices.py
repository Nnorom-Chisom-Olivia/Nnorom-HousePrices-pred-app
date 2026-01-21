import pandas as pd 
import numpy as np 
import joblib
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (mean_absolute_error, mean_squared_error, r2_score)
import os 

#Load the Dataset
data = pd.read_csv('train.csv')

#Preprocessing
print(data.head())
print(data.info())
columns_to_drop = ['Id', 'Alley', 'PoolQC', 'Fence', 'MiscFeature']
data = data.drop(columns=columns_to_drop, axis=1)
data = pd.get_dummies(data, drop_first=True)
data = data.fillna(data.median())

# We pick 6 from the list to make the Web GUI form manageable
selected_features = ['OverallQual', 'GrLivArea', 'TotalBsmtSF', 'GarageCars', 'FullBath', 'YearBuilt']
X = data[selected_features]
y = data['SalePrice']

# Split the Dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# LINEAR REGRESSION EVALUATION
model_lr = LinearRegression()
model_lr.fit(X_train, y_train)
y_pred_lr = model_lr.predict(X_test)

# Using your requested metrics for Linear Regression
mae_lr = mean_absolute_error(y_test, y_pred_lr)
mse_lr = mean_squared_error(y_test, y_pred_lr)
r2_lr = r2_score(y_test, y_pred_lr)

print("--- Linear Regression Performance ---")
print(f'MAE: {mae_lr}')
print(f'MSE: {mse_lr}')
print(f'R-squared: {r2_lr}')

# RANDOM FOREST EVALUATION
model_rf = RandomForestRegressor(n_estimators=100, random_state=42)
model_rf.fit(X_train, y_train)
y_pred_rf = model_rf.predict(X_test)

# Using metrics forto evaluate Random Forest
mae_rf = mean_absolute_error(y_test, y_pred_rf)
mse_rf = mean_squared_error(y_test, y_pred_rf)
r2_rf = r2_score(y_test, y_pred_rf)

print("\n--- Random Forest Performance ---")
print(f'MAE: {mae_rf}')
print(f'MSE: {mse_rf}')
print(f'R-squared: {r2_rf}')

#SCORAC SAVING REQUIREMENTS
if not os.path.exists('model'):
    os.makedirs('model')

# We save the Random Forest model because it performs better
joblib.dump(selected_features, "model/house_columns.joblib")
joblib.dump(model_rf, "model/house_price_model.pkl")