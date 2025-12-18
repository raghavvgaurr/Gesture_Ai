import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import joblib

DATA_DIR = "sign_data"

X, y = [], []

for file in os.listdir(DATA_DIR):
    if file.endswith(".csv"):
        label = file.replace(".csv", "")
        df = pd.read_csv(os.path.join(DATA_DIR, file))
        X.append(df)
        y += [label] * len(df)

X = pd.concat(X, ignore_index=True).to_numpy()
y = np.array(y)

le = LabelEncoder()
y_enc = le.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y_enc, test_size=0.2)

model = RandomForestClassifier(n_estimators=200)
model.fit(X_train, y_train)

accuracy = model.score(X_test, y_test)
print("Accuracy:", accuracy)

joblib.dump(model, "sign_rf_model.pkl")
joblib.dump(le, "sign_label_encoder.pkl")

print("Model saved!")

