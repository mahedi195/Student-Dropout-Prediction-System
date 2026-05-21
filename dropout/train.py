import os
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

import joblib

# ---------------------
# LOAD DATA
# ---------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(BASE_DIR)

file_path = os.path.join(PARENT_DIR, "student dropout.csv")
data = pd.read_csv(file_path)

# ---------------------
# FEATURES
# ---------------------
features = [
    'Study_Time',
    'Number_of_Failures',
    'Family_Support',
    'Wants_Higher_Education',
    'Number_of_Absences',
    'Grade_2',
    'Final_Grade'
]

df = data[features + ['Dropped_Out']]

# ---------------------
# ENCODING (FIXED POSITION)
# ---------------------
df = df.replace({
    'yes': 1,
    'no': 0,
    'Yes': 1,
    'No': 0
})

for col in df.columns:
    if df[col].dtype == 'object':
        df[col] = pd.Categorical(df[col]).codes

# ---------------------
# SPLIT DATA
# ---------------------
X = df[features]
y = df['Dropped_Out']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# ---------------------
# IMPUTER
# ---------------------
imputer = SimpleImputer(strategy='most_frequent')
X_train = imputer.fit_transform(X_train)
X_test = imputer.transform(X_test)

# ---------------------
# SCALING (FIXED)
# ---------------------
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ---------------------
# MODEL
# ---------------------
model = Sequential([
    Dense(32, activation='relu', input_shape=(X_train.shape[1],)),
    Dense(16, activation='relu'),
    Dense(8, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

model.fit(X_train, y_train, epochs=20, batch_size=16, verbose=1)

# ---------------------
# SAVE MODEL
# ---------------------
model.save("dropout_model.keras")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(imputer, "imputer.pkl")