import pandas as pd
import streamlit as st

from utils_folder.helper import load_pipeline
from src.features import add_engineered_features


pipeline = load_pipeline()

st.title("ASGMD05-HNW")
st.write("Enter passenger information below to predict whether the passenger was transported.")

PassengerId = st.text_input("PassengerId", value="0001_01")
HomePlanet = st.selectbox("HomePlanet", ["Earth", "Europa", "Mars"])
CryoSleep = st.selectbox("CryoSleep", [True, False])
Cabin = st.text_input("Cabin", value="B/0/P")
Destination = st.selectbox("Destination", ["TRAPPIST-1e", "PSO J318.5-22", "55 Cancri e"])
Age = st.number_input("Age", min_value=0.0, max_value=100.0, value=27.0)
VIP = st.selectbox("VIP", [False, True])
RoomService = st.number_input("RoomService", min_value=0.0, value=0.0)
FoodCourt = st.number_input("FoodCourt", min_value=0.0, value=0.0)
ShoppingMall = st.number_input("ShoppingMall", min_value=0.0, value=0.0)
Spa = st.number_input("Spa", min_value=0.0, value=0.0)
VRDeck = st.number_input("VRDeck", min_value=0.0, value=0.0)
Name = st.text_input("Name", value="Hazel Novero Winata")

if st.button("Predict"):
    input_df = pd.DataFrame(
        {
            "PassengerId": [PassengerId],
            "HomePlanet": [HomePlanet],
            "CryoSleep": [CryoSleep],
            "Cabin": [Cabin],
            "Destination": [Destination],
            "Age": [Age],
            "VIP": [VIP],
            "RoomService": [RoomService],
            "FoodCourt": [FoodCourt],
            "ShoppingMall": [ShoppingMall],
            "Spa": [Spa],
            "VRDeck": [VRDeck],
            "Name": [Name],
        }
    )

    input_df = add_engineered_features(input_df)
    prediction = pipeline.predict(input_df)[0]

    if prediction == 1:
        st.success("Prediction: Transported = True")
    else:
        st.error("Prediction: Transported = False")