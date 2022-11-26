import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import streamlit as st
CHOOSEN_MODEL = None
df = pd.read_csv('data-1.csv')
st.title("Linear Regression")
st.subheader("Data")
st.write(df)
st.subheader("Linear Regression")
x = df[['House Age', 'Distance from nearest Metro station (km)', 'Number of convenience stores',
         'House size (sqft)', 'Transaction date', 'Number of bedrooms']]
y = df['House price of unit area']
model = LinearRegression()
model.fit(x, y)
st.write("Coefficient of determination R^2 of the prediction:", model.score(x, y))
st.write("Mean squared error:", mean_squared_error(y, model.predict(x)))
st.subheader("Cross Validation")
score = cross_val_score(model, x, y, cv=5)
st.write("Cross Validation Score:", np.mean(score))
st.subheader("Residual Plot")
st.subheader("SVM")
from sklearn.svm import SVR
svr = SVR(kernel='rbf', C=100, gamma=0.1, epsilon=.1)
svr.fit(x, y)
st.write("Coefficient of determination R^2 of the prediction:", svr.score(x, y))
st.write("Mean squared error:", mean_squared_error(y, svr.predict(x)))
st.subheader("Neural Network")
from sklearn.neural_network import MLPRegressor
mlp = MLPRegressor(hidden_layer_sizes=(13, 13, 13), max_iter=500)
mlp.fit(x, y)
st.write("Coefficient of determination R^2 of the prediction:", mlp.score(x, y))
st.write("Mean squared error:", mean_squared_error(y, mlp.predict(x)))
st.subheader("Polynomial Regression")
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
model = make_pipeline(PolynomialFeatures(2), LinearRegression())
model.fit(x, y)
st.write("Coefficient of determination R^2 of the prediction:", model.score(x, y))
st.write("Mean squared error:", mean_squared_error(y, model.predict(x)))
st.subheader("Decision Tree")
from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(random_state=0)
regressor.fit(x, y)
st.write("Coefficient of determination R^2 of the prediction:", regressor.score(x, y))
st.write("Mean squared error:", mean_squared_error(y, regressor.predict(x)))
st.subheader("Random Forest")
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators=10, random_state=0)
regressor.fit(x, y)
st.write("Coefficient of determination R^2 of the prediction:", regressor.score(x, y))
st.write("Mean squared error:", mean_squared_error(y, regressor.predict(x)))
st.subheader("KNN")
from sklearn.neighbors import KNeighborsRegressor
regressor = KNeighborsRegressor(n_neighbors=5)
regressor.fit(x, y)
st.write("Coefficient of determination R^2 of the prediction:", regressor.score(x, y))
st.write("Mean squared error:", mean_squared_error(y, regressor.predict(x)))
st.subheader("Gradient Boosting")
from sklearn.ensemble import GradientBoostingRegressor
regressor = GradientBoostingRegressor(random_state=0)
regressor.fit(x, y)
st.write("Coefficient of determination R^2 of the prediction:", regressor.score(x, y))
st.write("Mean squared error:", mean_squared_error(y, regressor.predict(x)))
st.subheader("Ada Boost")
from sklearn.ensemble import AdaBoostRegressor
regressor = AdaBoostRegressor(random_state=0, n_estimators=100)
regressor.fit(x, y)
st.write("Coefficient of determination R^2 of the prediction:", regressor.score(x, y))
st.write("Mean squared error:", mean_squared_error(y, regressor.predict(x)))
st.subheader("XG Boost")
from xgboost import XGBRegressor
regressor = XGBRegressor()
regressor.fit(x, y)
st.write("Coefficient of determination R^2 of the prediction:", regressor.score(x, y))
st.write("Mean squared error:", mean_squared_error(y, regressor.predict(x)))
st.subheader("Light GBM")
from lightgbm import LGBMRegressor
regressor = LGBMRegressor()
regressor.fit(x, y)
st.write("Coefficient of determination R^2 of the prediction:", regressor.score(x, y))
st.write("Mean squared error:", mean_squared_error(y, regressor.predict(x)))
st.subheader("Cat Boost")
from catboost import CatBoostRegressor
regressor = CatBoostRegressor()
regressor.fit(x, y)
st.write("Coefficient of determination R^2 of the prediction:", regressor.score(x, y))
st.write("Mean squared error:", mean_squared_error(y, regressor.predict(x)))
st.subheader("Extra Trees")
from sklearn.ensemble import ExtraTreesRegressor
regressor = ExtraTreesRegressor(n_estimators=100, random_state=0)
regressor.fit(x, y)
st.write("Coefficient of determination R^2 of the prediction:", regressor.score(x, y))
st.write("Mean squared error:", mean_squared_error(y, regressor.predict(x)))
st.subheader("Prediction")
distance = st.number_input(
    "Distance from nearest Metro station (km)", min_value=0.0, max_value=100000.0, value=0.0)
number = st.number_input("Number of convenience stores",
                         min_value=0, max_value=10000, value=0, step=1)
house_age = st.number_input(
    "House Age", min_value=0.0, max_value=100000.0, value=0.0)

no_bedrooms = st.number_input(
    "Number of bedrooms", min_value=0, max_value=10000, value=0, step=1)
house_size = st.number_input(
    "House size (sqft)", min_value=0, max_value=10000, value=0, step=1)
transaction_date = st.number_input(
    "Transaction date", min_value=0, max_value=10000000, value=0, step=1)
if st.button("Predict"):
    output = svr.predict(
        [[house_age, distance, number,  house_size, transaction_date, no_bedrooms, ]])
    st.success('The House price of unit area is {}'.format(output))