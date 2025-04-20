
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier

# จำลองข้อมูล
np.random.seed(42)
rounds = 2000
results = np.random.choice(['P', 'B', 'T'], size=rounds, p=[0.45, 0.45, 0.10])
df = pd.DataFrame({'result': results})
result_map = {'P': 0, 'B': 1, 'T': 2}
df['result_code'] = df['result'].map(result_map)

for i in range(1, 6):
    df[f'prev_{i}'] = df['result_code'].shift(i)

def get_streak(data):
    streaks = []
    current = None
    count = 0
    for value in data:
        if value == current:
            count += 1
        else:
            count = 1
            current = value
        streaks.append(count)
    return streaks

df['streak'] = get_streak(df['result_code'])

def rolling_counts(series, code):
    return (series == code).astype(int).rolling(window=5).sum()

df['P_count_5'] = rolling_counts(df['result_code'], 0)
df['B_count_5'] = rolling_counts(df['result_code'], 1)
df['T_count_5'] = rolling_counts(df['result_code'], 2)

df.dropna(inplace=True)

features = ['prev_1', 'prev_2', 'prev_3', 'prev_4', 'prev_5', 'streak', 'P_count_5', 'B_count_5', 'T_count_5']
X = df[features]
y = df['result_code']

model = GradientBoostingClassifier(n_estimators=200, learning_rate=0.1, max_depth=4)
model.fit(X, y)

# UI
st.title("Advanced Baccarat Predictor")
st.write("ทำนายผลบาคาร่าโดยใช้รูปแบบย้อนหลัง พร้อมฟีเจอร์ขั้นสูง")

option_map = {"Player (P)": 0, "Banker (B)": 1, "Tie (T)": 2}

cols = st.columns(5)
inputs = [cols[i].selectbox(f"ผลย้อนหลัง {i+1}", list(option_map.keys()), index=0) for i in range(5)]

streak = st.number_input("จำนวน streak ล่าสุด (เช่น B ชนะติดกันกี่ตา)", min_value=1, max_value=20, value=1)
p_count = st.number_input("จำนวนครั้งที่ P ออกใน 5 ตาหลัง", min_value=0, max_value=5, value=2)
b_count = st.number_input("จำนวนครั้งที่ B ออกใน 5 ตาหลัง", min_value=0, max_value=5, value=2)
t_count = st.number_input("จำนวนครั้งที่ T ออกใน 5 ตาหลัง", min_value=0, max_value=5, value=1)

if st.button("ทำนายผล"):
    input_data = [[option_map[i] for i in inputs] + [streak, p_count, b_count, t_count]]
    result = model.predict(input_data)[0]
    result_decode = {0: "Player (P)", 1: "Banker (B)", 2: "Tie (T)"}
    st.success(f"ระบบคาดการณ์ว่า: **{result_decode[result]}**")
