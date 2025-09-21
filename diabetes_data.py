import numpy as np
import pandas as pd
import random
from tqdm import tqdm

np.random.seed(42)
random.seed(42)

N_PATIENTS = 10
DAYS = 180

# Generate patient base
def generate_patient_base(n):
    patient_ids = [f"P{str(i).zfill(5)}" for i in range(1, n+1)]
    sex = np.random.choice(['male', 'female'], size=n)
    age = np.random.randint(5, 71, size=n)
    education = np.random.choice([1,2,3,4,5,6], size=n, p=[0.1,0.15,0.25,0.25,0.15,0.1])
    income = np.random.choice([1,2,3,4,5,6,7,8], size=n, p=[0.1,0.15,0.2,0.2,0.15,0.1,0.05,0.05])
    return pd.DataFrame({
        'patient_id': patient_ids,
        'Sex': sex,
        'Age': age,
        'Education': education,
        'Income': income
    })

# Simulate a day's readings for a patient (tuned to LLM sample)
def simulate_day_row(patient):
    age = patient['Age']
    bmi = np.clip(np.random.normal(27 + (age-40)/25, 5), 16, 45)  # LLM sample: slightly higher mean, more spread
    highbp = int(bmi > 29 or age > 55 or random.random() < 0.13)
    highchol = int(bmi > 27 or age > 50 or random.random() < 0.15)
    cholcheck = int(random.random() < 0.97)
    smoker = int(random.random() < (0.13 if age < 18 else 0.22))  # LLM: slightly lower
    stroke = int(highbp and random.random() < 0.025)
    heart = int(highbp and highchol and random.random() < 0.045)
    physact = int(random.random() < 0.68)
    fruits = int(random.random() < 0.58)
    veggies = int(random.random() < 0.68)
    hvyalc = int(random.random() < 0.07)
    anyhc = int(random.random() < 0.99)
    nodoccost = int(random.random() < 0.04)
    genhlth = np.random.choice([1,2,3,4,5], p=[0.13,0.27,0.32,0.18,0.10])
    menthlth = np.clip(int(np.random.normal(2.5, 2.5)), 0, 30)
    physhlth = np.clip(int(np.random.normal(3.5, 3.5)), 0, 30)
    diffwalk = int(age > 62 or bmi > 36 or random.random() < 0.06)
    return [highbp, highchol, cholcheck, round(bmi,1), smoker, stroke, heart, physact, fruits, veggies, hvyalc, anyhc, nodoccost, genhlth, menthlth, physhlth, diffwalk]

# Assign deplict label based on risk factors in 180 days (tuned for LLM-like prevalence)
def assign_deplict_label(df):
    highbp = df['HighBP'].mean() > 0.55
    highchol = df['HighChol'].mean() > 0.55
    highbmi = (df['BMI'] > 30).mean() > 0.35
    smoker = df['Smoker'].mean() > 0.22
    return int(highbp or highchol or highbmi or smoker)

# Main generation
patients = generate_patient_base(N_PATIENTS)
daily_columns = [
    'HighBP', 'HighChol', 'CholCheck', 'BMI', 'Smoker', 'Stroke', 'HeartDiseaseorAttack',
    'PhysActivity', 'Fruits', 'Veggies', 'HvyAlcoholConsump', 'AnyHealthcare', 'NoDocbcCost',
    'GenHlth', 'MentHlth', 'PhysHlth', 'DiffWalk'
]

all_rows = []
for idx, patient in tqdm(patients.iterrows(), total=N_PATIENTS):
    patient_days = []
    for day in range(1, DAYS+1):
        row = simulate_day_row(patient)
        patient_days.append(row)
    df_days = pd.DataFrame(patient_days, columns=daily_columns)
    deplict = assign_deplict_label(df_days)
    df_days['patient_id'] = patient['patient_id']
    df_days['day'] = np.arange(1, DAYS+1)
    df_days['Sex'] = patient['Sex']
    df_days['Age'] = patient['Age']
    df_days['Education'] = patient['Education']
    df_days['Income'] = patient['Income']
    df_days['deplict'] = deplict
    all_rows.append(df_days)

df_final = pd.concat(all_rows, ignore_index=True)
cols = ['patient_id', 'day'] + daily_columns + ['Sex', 'Age', 'Education', 'Income', 'deplict']
df_final = df_final[cols]

# Save to CSV
df_final.to_csv('test_data.csv', index=False)
print('Done! File saved as patient_180days_readings.csv')