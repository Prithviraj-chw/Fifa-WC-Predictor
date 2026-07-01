# ⚽ FIFA World Cup Predictor

A machine learning web app that predicts the outcome of FIFA World Cup matches using historical match data and FIFA rankings. Built with classical ML

🔗 **[Live App](https://fifa-wc-predictor-gz4pzhkfecgoappep8rvjnf.streamlit.app/)**

---

## 📌 Project Overview

Given two national teams, the app predicts:
- **Match outcome** — Home Win / Draw / Away Win
- **Win probabilities** for each outcome

The model is trained on 15,000+ competitive international matches from 1993 to 2024, including World Cup, UEFA Euro, Copa América, Africa Cup of Nations, and qualification matches.

---

## 🧠 Models Used

| Model | Accuracy |
|---|---|
| Logistic Regression | 59.2% |
| Random Forest | 59.2% |
| **XGBoost (deployed)** | **59.4%** |

> ⚠️ A ~59% accuracy is strong for football prediction. The theoretical ceiling for this problem is 55–65% — even professional analysts and betting companies rarely exceed this due to the inherent unpredictability of football.

All three models were tuned using `RandomizedSearchCV` with 5-fold cross validation.

---

## 🔧 Features Used

| Feature | Description |
|---|---|
| `home_rank` | FIFA ranking of home team at match date |
| `away_rank` | FIFA ranking of away team at match date |
| `rank_diff` | Difference in FIFA rankings |
| `home_points` | FIFA points of home team |
| `away_points` | FIFA points of away team |
| `points_diff` | Difference in FIFA points |
| `home_win_rate` | Home team win rate in last 5 matches |
| `home_goal_avg` | Home team average goals scored in last 5 matches |
| `home_conceded_avg` | Home team average goals conceded in last 5 matches |
| `away_win_rate` | Away team win rate in last 5 matches |
| `away_goal_avg` | Away team average goals scored in last 5 matches |
| `away_conceded_avg` | Away team average goals conceded in last 5 matches |
| `is_neutral` | Whether the match is played on neutral ground |

---

## 📂 Project Structure

```
fifa-world-cup-predictor/
├── data/
│   ├── results.csv                    ← International match results (1872–2024)
│   └── fifa_ranking-2024-06-20.csv   ← FIFA world rankings (1992–2024)
├── models/
│   ├── XGBoost.pkl                   ← Trained XGBoost model
│   └── scaler.pkl                    ← StandardScaler
├── model.ipynb                       ← Data prep, feature engineering, training
├── predictor_app.py                  ← Streamlit frontend
└── requirements.txt
```

---

## 📊 Datasets

| Dataset | Source |
|---|---|
| International Football Results (1872–2024) | [Kaggle](https://www.kaggle.com/datasets/martj42/international-football-results-from-1872-to-2017) |
| FIFA World Rankings (1992–2024) | [Kaggle](https://www.kaggle.com/datasets/cashncarry/fifaworldranking) |

---

## ⚙️ How It Works

1. **Data Collection** — Two Kaggle datasets: match results and FIFA rankings
2. **Filtering** — Kept only competitive matches (World Cup, Euros, Copa América etc.) — ~18,000 rows
3. **Merging** — Attached each team's FIFA ranking at the time of every match using `pd.merge_asof`
4. **Feature Engineering** — Computed rolling form features (win rate, goals avg, concede avg) over last 5 matches per team
5. **Training** — Time-based split (pre-2018 train, 2018+ test), tuned with `RandomizedSearchCV`
6. **Deployment** — Streamlit frontend deployed on Streamlit Community Cloud

---

## 🚀 Run Locally

```bash
# clone the repo
git clone https://github.com/Prithviraj-chw/fifa-world-cup-predictor
cd fifa-world-cup-predictor

# install dependencies
pip install -r requirements.txt

# run the app
streamlit run predictor_app.py
```

---

## 🛠️ Tech Stack

- **Python 3.11**
- **pandas** — data manipulation and merging
- **scikit-learn** — Logistic Regression, Random Forest, preprocessing, CV
- **XGBoost** — final deployed model
- **Streamlit** — frontend and deployment
- **joblib** — model serialization

---

## 📈 Key Design Decisions

- **Time-based train/test split** instead of random split — prevents data leakage from future matches
- **RandomizedSearchCV** instead of GridSearchCV — 5–10x faster hyperparameter tuning on CPU
- **Classical ML only** — XGBoost outperforms deep learning on structured tabular match data
- **Competitive matches only** — excluded friendlies since teams don't play seriously in them

---

## 👤 Author

**Prithviraj Chowdhury**

