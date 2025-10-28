import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
import joblib

# --------------------------
# LOAD DATASETS
# --------------------------
print("Loading datasets...")

delivery_df = pd.read_csv("dataset/delivery_performance.csv")
orders_df = pd.read_csv("dataset/orders.csv")
routes_df = pd.read_csv("dataset/routes_distance.csv")

# --------------------------
# MERGE DATASETS
# --------------------------
print("Merging datasets...")
df = (
    delivery_df.merge(orders_df, on="Order_ID", how="left")
    .merge(routes_df, on="Order_ID", how="left")
)

# --------------------------
# DEFINE TARGET VARIABLE
# --------------------------
# Delivery_Flag = 1 if delayed, else 0
df["Delay_Flag"] = (df["Actual_Delivery_Days"] > df["Promised_Delivery_Days"]).astype(int)

# --------------------------
# SELECT FEATURES
# --------------------------
features = ["Carrier", "Priority", "Promised_Delivery_Days", "Route", "Weather_Impact"]
target = "Delay_Flag"

df = df[features + [target]].dropna()

print(f"Using {len(features)} features for training: {features}")
print(f"Data shape after cleanup: {df.shape}")

# --------------------------
# TRAIN/TEST SPLIT
# --------------------------
X_train, X_test, y_train, y_test = train_test_split(
    df[features], df[target], test_size=0.2, random_state=42, stratify=df[target]
)

# --------------------------
# MODEL PIPELINE
# --------------------------
categorical_features = ["Carrier", "Priority", "Route", "Weather_Impact"]
numeric_features = ["Promised_Delivery_Days"]

preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ("num", "passthrough", numeric_features),
    ]
)

model = XGBClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=4,
    subsample=0.9,
    colsample_bytree=0.9,
    random_state=42,
    eval_metric="logloss",
    use_label_encoder=False
)

pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])

# --------------------------
# TRAIN MODEL
# --------------------------
print("Training model...")
pipeline.fit(X_train, y_train)
print("Model training completed!")

# --------------------------
# EVALUATE
# --------------------------
train_acc = pipeline.score(X_train, y_train)
test_acc = pipeline.score(X_test, y_test)
print(f"Training Accuracy: {train_acc:.2%}")
print(f"Test Accuracy: {test_acc:.2%}")

# --------------------------
# SAVE MODEL
# --------------------------
os.makedirs("models", exist_ok=True)
joblib.dump(pipeline, "models/delay_predictor.pkl")
print("Model saved to models/delay_predictor.pkl")

print("Predictor ready for use in app.py")
