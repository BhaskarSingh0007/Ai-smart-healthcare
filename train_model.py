import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
import joblib
import os

# 1. LOAD DATASET
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
columns = ["age","sex","cp","trestbps","chol","fbs","restecg","thalach",
           "exang","oldpeak","slope","ca","thal","target"]
df = pd.read_csv(url, names=columns, na_values="?")
print("Dataset Loaded | Shape:", df.shape)

# 2. HANDLE MISSING VALUES
imputer = SimpleImputer(strategy='median')
df[df.columns] = imputer.fit_transform(df)

# 3. BINARY TARGET
df["target"] = df["target"].apply(lambda x: 1 if x > 0 else 0)

# 4. SPLIT
X = df.drop("target", axis=1)
y = df["target"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42)

# 5. PIPELINE & TUNING
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model", RandomForestClassifier(random_state=42))
])
param_grid = {
    "model__n_estimators": [100, 200],
    "model__max_depth": [4, 6, 8]
}
print("Training model... please wait...")
grid = GridSearchCV(pipeline, param_grid, cv=3, scoring="accuracy")
grid.fit(X_train, y_train)
rf_model = grid.best_estimator_
print("Best Parameters:", grid.best_params_)

# 6. EVALUATE
y_pred = rf_model.predict(X_test)
y_prob = rf_model.predict_proba(X_test)[:, 1]
print("\n=== Classification Report ===")
print(classification_report(y_test, y_pred))
print("AUC Score:", round(roc_auc_score(y_test, y_prob), 4))

# 7. SAVE MODEL
save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                         "rf_heart_model.pkl")
joblib.dump(rf_model, save_path)
print(f"\nSUCCESS: Model saved as rf_heart_model.pkl")
print("Now run: python main.py")