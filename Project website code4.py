pip install gradio
import gradio as gr
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.impute import SimpleImputer
import pickle

# Load the trained model (make sure to save it as 'car_price_model.pkl' before)
with open("car_price_model.pkl", "rb") as file:
    model_pipeline = pickle.load(file)

# Define feature names
categorical_cols = ['name', 'company', 'fuel_type']
numerical_cols = ['kms_driven', 'year']

# Function to predict car price
def predict_price(name, company, year, kms_driven, fuel_type):
    # Create input dataframe
    input_data = pd.DataFrame([[name, company, year, kms_driven, fuel_type]],
                              columns=['name', 'company', 'year', 'kms_driven', 'fuel_type'])

    # Predict price
    price_pred_log = model_pipeline.predict(input_data)
    price_pred = np.expm1(price_pred_log)  # Convert back from log scale
    return f"Predicted Price: ₹{price_pred[0]:,.2f}"

# Create Gradio interface
inputs = [
    gr.Textbox(label="Car Name"),
    gr.Textbox(label="Company"),
    gr.Number(label="Year", value=2015),
    gr.Number(label="Kilometers Driven", value=50000),
    gr.Dropdown(choices=['Petrol', 'Diesel', 'CNG', 'Electric'], label="Fuel Type")
]

output = gr.Textbox(label="Estimated Price")

gr.Interface(fn=predict_price, inputs=inputs, outputs=output, title="Car Price Prediction", live=True).launch()
