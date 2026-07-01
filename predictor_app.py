import streamlit as st
import joblib
import pandas as pd
st.title("⚽ Fifa world Cup Predictor")
st.header("Select two teams to predict their match outcome")

model = joblib.load("XGBoost.pkl")
scaler = joblib.load("scaler.pkl")

teams = sorted([
    'Brazil', 'Argentina', 'France', 'Germany', 'Spain',
    'England', 'Portugal', 'Netherlands', 'Italy', 'Belgium',
    'Uruguay', 'Croatia', 'Denmark', 'Mexico', 'USA',
    'Senegal', 'Morocco', 'Japan', 'Korea Republic', 'Australia',
    'Poland', 'Switzerland', 'Ghana', 'Cameroon', 'Serbia',
    'Ecuador', 'Qatar', 'Wales', 'Tunisia', 'Costa Rica',
    'Canada', 'Saudi Arabia', 'Iran'
])

col1,col2 = st.columns(2) #splits the page into two columns (sections), our 2 dropdowns can be present adjacently instead of vertically
with col1:
    home_team = st.selectbox("Select home team", teams)
with col2:
    away_team = st.selectbox("Select away team", teams)

rankings = pd.read_csv("fifa_ranking-2024-06-20.csv")
rankings["rank_date"] = pd.to_datetime(rankings["rank_date"])


