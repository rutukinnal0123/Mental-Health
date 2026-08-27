# train.py

import pandas as pd
import joblib

from sklearn.model_selection import train_test_split

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    average_precision_score
)

from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier

from src.preprocess import preprocessor


# ============================================================
# 1. LOAD DATA
# ============================================================

df = pd.read_csv("data/mental_health.csv")


# ============================================================
# 2. CLEAN CATEGORICAL TEXT
# ============================================================

categorical_columns = df.select_dtypes(
    include=["object", "string", "category"]
).columns

for column in categorical_columns:
    df[column] = (
        df[column]
        .str.strip()
        .str.lower()
    )


# ============================================================
# 3. DEFINE TARGET AND REMOVE LEAKAGE
# ============================================================

target = "Has_Mental_Health_Issue"

leakage_columns = [
    "Previously_Diagnosed",
    "Ever_Sought_Treatment",
    "On_Therapy_Now",
    "On_Medication"
]

# Features
X = df.drop(
    columns=[target] + leakage_columns
)

# Target
y = df[target]


print("X shape:", X.shape)
print("y shape:", y.shape)


# ============================================================
# 4. CHECK CLASS DISTRIBUTION
# ============================================================

print("\nOverall class distribution:")
print(y.value_counts())

print("\nOverall class percentage:")
print(y.value_counts(normalize=True) * 100)


# ============================================================
# 5. TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


print("\nTraining set:", X_train.shape)
print("Testing set:", X_test.shape)

print("\nTraining class distribution:")
print(y_train.value_counts())

print("\nTesting class distribution:")
print(y_test.value_counts())


# ============================================================
# 6. FIT PREPROCESSOR ONLY ON TRAINING DATA
# ============================================================

X_train_processed = preprocessor.fit_transform(
    X_train
)

X_test_processed = preprocessor.transform(
    X_test
)


print(
    "\nProcessed training shape:",
    X_train_processed.shape
)

print(
    "Processed testing shape:",
    X_test_processed.shape
)


# ============================================================
# 7. APPLY SMOTE ONLY TO TRAINING DATA
# ============================================================

smote = SMOTE(
    random_state=42
)

X_train_resampled, y_train_resampled = smote.fit_resample(
    X_train_processed,
    y_train
)


print("\nSMOTE applied!")

print("Before SMOTE:")
print(y_train.value_counts())

print("\nAfter SMOTE:")
print(y_train_resampled.value_counts())

print(
    "\nResampled training shape:",
    X_train_resampled.shape
)


# ============================================================
# 8. SAVE PREPROCESSOR
# ============================================================

joblib.dump(
    preprocessor,
    "models/preprocessor.pkl"
)

print("\nPreprocessor saved!")


# ============================================================
# 9. TRAIN XGBOOST
# ============================================================

model = XGBClassifier(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=5,
    min_child_weight=3,
    subsample=0.8,
    colsample_bytree=0.8,

    objective="binary:logistic",

    # PR-AUC is useful for imbalanced classification
    eval_metric="aucpr",

    random_state=42,
    n_jobs=-1
)


# Train on SMOTE-resampled data
model.fit(
    X_train_resampled,
    y_train_resampled
)


print("\nXGBoost model training completed!")


# ============================================================
# 10. SAVE MODEL
# ============================================================

joblib.dump(
    model,
    "models/xgboost_model.pkl"
)

print("XGBoost model saved!")


# ============================================================
# 11. MAKE PREDICTIONS
# ============================================================

# Predicted class
y_pred = model.predict(
    X_test_processed
)

# Probability for class 1
y_prob = model.predict_proba(
    X_test_processed
)[:, 1]


print("\nPredictions completed!")


# ============================================================
# 12. ACCURACY
# ============================================================

accuracy = accuracy_score(
    y_test,
    y_pred
)

print("\nAccuracy:", accuracy)


# ============================================================
# 13. CONFUSION MATRIX
# ============================================================

print("\nConfusion Matrix:")

print(
    confusion_matrix(
        y_test,
        y_pred
    )
)


# ============================================================
# 14. CLASSIFICATION REPORT
# ============================================================

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        y_pred
    )
)


# ============================================================
# 15. ROC-AUC
# ============================================================

roc_auc = roc_auc_score(
    y_test,
    y_prob
)

print("\nROC-AUC:", roc_auc)


# ============================================================
# 16. PR-AUC
# ============================================================

pr_auc = average_precision_score(
    y_test,
    y_prob
)

print("PR-AUC:", pr_auc)