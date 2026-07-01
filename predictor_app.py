import streamlit as st
import joblib
import pandas as pd
#apply font colour
st.markdown("""
    <style>
    /* title */
    .stApp h1 {
        color: #C9A84C !important;
    }

    /* subheaders */
    .stApp h2, .stApp h3 {
        color: #63c5b7 !important; /* !important forces our color to override streamlit default theme styles*/
    }

    /* regular text */
    .stApp p {
        color: #e7edf4 !important;
    }

    /* try all possible label selectors for dropdown label color */
    div[data-testid="stSelectbox"] label,
    div[data-testid="stSelectbox"] p,
    div[data-baseweb="select"] label,
    .stSelectbox label p {
        color: #e08e2e !important;
    }
    </style>
""", unsafe_allow_html=True)
st.set_page_config(
    page_title="Predict the WC matches",
    page_icon="⚽",
    layout="centered"
)
st.title("Fifa world Cup Predictor")
st.header("Select two teams to predict their match outcome")

model = joblib.load("XGBoost.pkl")
scaler = joblib.load("scaler.pkl")

teams = sorted([
    'Argentina', 'France', 'Brazil', 'England', 'Portugal',
    'Spain', 'Netherlands', 'Germany', 'Belgium', 'Croatia',
    'Uruguay', 'Denmark', 'Switzerland', 'Japan', 'Korea Republic',
    'Morocco', 'Senegal', 'USA', 'Mexico', 'Canada',
    'Australia', 'Poland', 'Serbia', 'Colombia', 'Ecuador',
    'Peru', 'Chile', 'Venezuela', 'Bolivia', 'Paraguay',
    'Panama', 'Costa Rica', 'Honduras', 'El Salvador', 'Guatemala',
    'Jamaica', 'Cuba', 'Trinidad and Tobago',
    'Nigeria', 'Ghana', 'Cameroon', "Côte d'Ivoire", 'Mali',
    'Egypt', 'Algeria', 'Tunisia', 'Morocco', 'South Africa',
    'Saudi Arabia', 'IR Iran', 'Japan', 'Korea Republic',
    'Australia', 'Qatar', 'Iraq', 'Jordan', 'Uzbekistan',
    'China PR', 'Indonesia', 'New Zealand',
    'Italy', 'Austria', 'Hungary', 'Czechia', 'Slovakia',
    'Romania', 'Turkey', 'Georgia', 'Scotland', 'Norway',
    'Sweden', 'Ukraine', 'Slovenia', 'Republic of Ireland',
    'Wales', 'Northern Ireland', 'Greece',
])

# remove duplicates from the sorted list
teams = sorted(list(set(teams)))

col1,col2 = st.columns(2) #splits the page into two columns (sections), our 2 dropdowns can be present adjacently instead of vertically
with col1:
    home_team = st.selectbox("🏠 Select home team", teams)
with col2:
    away_team = st.selectbox("✈️ Select away team", teams)

rankings = pd.read_csv("fifa_ranking-2024-06-20.csv")
rankings["rank_date"] = pd.to_datetime(rankings["rank_date"])

#center the predict button and make its size a bit larger
st.markdown("""
    <style>
    div.stButton > button { 
        width: 300px;
        height: 50px;
        font-size: 30px;
    }
    /* button */
div.stButton > button {
    background-color: #934018 !important;
    color: #000000 !important;
    border: none !important;
}

/* button on hover */
div.stButton > button:hover {
    background-color: #6b2e10 !important;
    color: #000000 !important;
}
    </style>
""", unsafe_allow_html=True)

col_left,col_mid,col_right = st.columns([1.5,2,1.5]) #left col takes 1 part, middle takes 2 and right takes 1
with col_mid:
    predict = st.button("Predict the match outcome",use_container_width=True) #use_container_width=True makes the button fill the full width of that middle column so it looks wide and prominent.
if predict:
    home_ranking = rankings[rankings["country_full"] == home_team].iloc[-1] #filters to the selected home team row and [-1] filters to the last entry
    away_ranking = rankings[rankings["country_full"] == away_team].iloc[-1]

    home_rank = home_ranking["rank"]
    away_rank = away_ranking["rank"]
    home_pts = home_ranking["total_points"]
    away_pts = away_ranking["total_points"]

    rank_diff = home_rank - away_rank
    points_diff = home_pts - away_pts

    features = pd.DataFrame([[home_rank,away_rank,rank_diff,points_diff, #features same as our training features of model
                              1, #1 for neutral ground match
                              0.5, 1.2, 0.8,  # home team defaults
                              0.5, 1.0, 0.9 #away team defaults
                              ]], columns=[
    'home_rank', 'away_rank', 'rank_diff',
    'points_diff', 'is_neutral',
    'home_win_rate', 'home_avg_goals', 'home_goals_conceded',
    'away_win_rate', 'away_avg_goals', 'away_goals_conceded',
    ])

    #scale and transform features
    features_scaled = scaler.transform(features)
    predictions = model.predict(features_scaled)[0] #returns smth like [2] , always a single digit array, [0] just takes in the nos
    probabilities = model.predict_proba(features_scaled)[0] #it would return a 2d array [0] converts it into a 1d array

    if predictions == 2:
        st.success(f"{home_team} wins") #green box to display the msg
    elif predictions == 0:
        st.warning(f"{away_team} wins") #yellow box to display the msg
    else:
        st.info("Its a draw") #blue box displays the msg

    st.write("Win probabilities")
    probab_df = pd.DataFrame({
        "Outcome": [f"{home_team} win","draw", f"{away_team} win"],
        "Probabilites": [f"{probabilities[2]:.0%}",f"{probabilities[1]:.0%}", f"{probabilities[0]:.0%}"],
    })
    st.dataframe(probab_df,hide_index=True)







