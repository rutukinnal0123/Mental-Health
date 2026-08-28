# Mental Health — Exploratory Data Analysis & Predictive Modeling

## Project Overview

This project analyzes a mental-health dataset containing **10,000 records and 51 columns** to understand the data, identify patterns, investigate the target variable, and evaluate whether the available features are suitable for reliable machine-learning prediction.

The project follows an end-to-end data-science workflow:

**Data Cleaning → EDA → Feature Analysis → Preprocessing → Class-Imbalance Handling → Model Training → Evaluation → Final Feasibility Assessment**

The main finding of the project is that the dataset is useful for **EDA, preprocessing, and understanding the data-science workflow**, but the available features provide **weak predictive separation between the target classes**, making reliable classification difficult.

---

## Problem Statement

The target variable is:

```text
Has_Mental_Health_Issue
```

The goal was to investigate whether the available demographic, lifestyle, work-related, behavioral, and mental-health-related features can reliably predict whether a person belongs to class `0` or class `1`.

---

## Dataset

The dataset contains:

* **10,000 rows**
* **51 columns**
* Numerical and categorical features
* No missing values
* No duplicate rows after cleaning
---

## Project Architecture

```text
                    ┌─────────────────────┐
                    │   mental_health.csv │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │   Data Cleaning      │
                    │                     │
                    │ • Missing values    │
                    │ • Duplicate checks  │
                    │ • Text normalization│
                    │ • Data inspection    │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │        EDA          │
                    │                     │
                    │ • Target analysis   │
                    │ • Histograms        │
                    │ • Count plots       │
                    │ • Boxplots          │
                    │ • Correlation       │
                    │ • Categorical       │
                    │   comparisons       │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Feature Preparation │
                    │                     │
                    │ • Encode categories │
                    │ • Remove leakage    │
                    │ • Train/test split  │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │   Preprocessing     │
                    │                     │
                    │ • Fit on train only │
                    │ • Transform train   │
                    │ • Transform test    │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Imbalance Handling  │
                    │                     │
                    │ • Class weighting   │
                    │ • SMOTE experiment  │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │      XGBoost        │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │    Evaluation       │
                    │                     │
                    │ • Accuracy          │
                    │ • Precision         │
                    │ • Recall            │
                    │ • F1-score          │
                    │ • Confusion matrix  │
                    │ • ROC-AUC           │
                    │ • PR-AUC            │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Final Feasibility   │
                    │      Assessment     │
                    └─────────────────────┘
```

---

# 1. Data Cleaning

The first phase focused on understanding and cleaning the dataset.

### Performed

* Loaded the dataset using Pandas
* Inspected the first rows
* Checked dataset shape
* Inspected data types
* Checked missing values
* Checked duplicate rows
* Identified categorical and numerical columns
* Cleaned categorical text using `.strip()` and `.lower()`
* Checked unique categorical values
* Checked category frequencies
* Investigated suspicious numerical ranges

### Result

The dataset contained:

* **0 missing values**
* **0 duplicate rows**

## and all columns contained 10,000 non-null observations.

# 2. Target Analysis

The target variable is:

```text
Has_Mental_Health_Issue
```

The class distribution is highly imbalanced:

| Class | Count | Percentage |
| ----- | ----: | ---------: |
| 1     | 9,216 |     92.16% |
| 0     |   784 |      7.84% |

This became one of the main challenges in the modeling phase.

### Why this matters

A model can obtain high accuracy simply by predicting the majority class most of the time while performing very poorly on the minority class.

Therefore, accuracy alone is not sufficient for evaluating this problem.

---

# 3. Exploratory Data Analysis

The EDA phase was used to understand how the features behave before modeling.

## Numerical Feature Analysis

Different visualization techniques were used depending on the type of numerical variable.

### Histograms

Used to understand:

* Distribution
* Concentration of values
* Skewness
* Possible unusual values
* Overall shape

Examples:

* `Age`
* `Work_Hours_Per_Week`
* `Sleep_Hours_Night`
* `Screen_Time_Hours_Day`
* `Social_Media_Hours_Day`

### Count Plots

Used for discrete variables where the actual values represent categories or small counts.

Examples include:

* Binary variables such as `Ever_Bullied_At_Work`
* Other small discrete variables such as `Panic_Attacks`

This avoids treating values such as `0` and `1` as if intermediate values like `0.2`, `0.4`, or `0.6` had meaningful interpretations.

### Boxplots

Used to examine:

* Median
* Spread
* Interquartile range
* Potential outliers

They were applied to numerical/distribution-oriented variables rather than binary/count variables.

The notebook explicitly separates binary variables for count plots and uses histograms for the remaining numerical variables.

---

# 4. Correlation Analysis

The numerical features were compared with the target using correlation.

The strongest listed numerical relationship was:

```text
Work_Stress_Level → 0.055
```

Other examples:

```text
Family_History_Mental_Illness → 0.0547
Feeling_Sad_Down              → 0.0501
Financial_Stress              → 0.0458
Anxious_Nervous               → 0.0436
```

while many features were close to zero.

### Interpretation

These results indicate **very weak linear relationships between individual numerical features and the target**.

However, correlation alone cannot prove that a feature contains no predictive information because it does not capture every possible nonlinear relationship or feature interaction.

---

# 5. Class-wise Feature Analysis

To investigate whether the two target classes were distinguishable, selected features were compared against the target using boxplots.

The notebook used:

```text
Work_Stress_Level
Feeling_Sad_Down
Social_Support
```

for class-wise comparisons.

The overall analysis suggested considerable overlap between the feature distributions of class 0 and class 1.

This means that people belonging to different target classes can have very similar feature values.

That creates a difficult classification problem because the model may not have a clear decision boundary.

---

# 6. Categorical Feature Analysis

Categorical features were compared against the target using normalized cross-tabulations.

The analysis showed that target percentages were often very similar across categories.

For example, for `Gender`:

| Gender            | Class 0 | Class 1 |
| ----------------- | ------: | ------: |
| Female            |   8.46% |  91.54% |
| Male              |   7.14% |  92.86% |
| Non-binary        |   8.54% |  91.46% |
| Prefer not to say |   7.76% |  92.24% |

Similar patterns were observed across features such as education, income, employment, remote work, exercise, alcohol frequency, smoking, diet quality, and discussion of mental health.

### Interpretation

The categorical variables also did not show large differences between the target classes.

---

# 7. Feature Preparation

For the modeling stage:

* The target was separated from the features.
* Several features were removed because they could introduce target leakage.
* Categorical variables were prepared for machine learning.
* Training and testing data were separated using a stratified split.

The target was kept separate from the feature matrix, and leakage-related variables such as `Previously_Diagnosed`, `On_Therapy_Now`, and `On_Medication` were removed from the predictive feature set.

---

# 8. Categorical Encoding

Categorical features were handled based on their type.

### Ordinal categories

Features with meaningful order were mapped to numerical values.

Examples:

* `Income_Level`
* `Exercise_Per_Week`
* `Alcohol_Frequency`
* `Diet_Quality`
* `Discuss_Mental_Health`

### Nominal categories

Features without a natural order were one-hot encoded.

Examples:

* `Gender`
* `Country`
* `Education`
* `Marital_Status`
* `Employment_Status`
* `Remote_Work`
* `Company_Mental_Health_Support`
* `Smoking`

This resulted in an expanded feature representation for machine learning.

---

# 9. Train/Test Split

The data was divided into training and testing sets using:

```python
train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)
```

`stratify=y` was used so that both training and testing datasets retained approximately the same class proportions.

---

# 10. Preprocessing

The preprocessing pipeline was fitted **only on the training data** and then used to transform the test data.

This prevents information from the test set from leaking into the training process.

The processed training data contained **63 features** in the final modeling pipeline.

---

# 11. Model Used

The main predictive model was **XGBoost**.

Configuration included:

```text
n_estimators = 300
learning_rate = 0.05
max_depth = 5
min_child_weight = 3
subsample = 0.8
colsample_bytree = 0.8
objective = binary:logistic
eval_metric = aucpr
```

XGBoost was selected because it is a powerful tree-based algorithm for structured/tabular data and can model nonlinear relationships and feature interactions.

---

# 12. Handling Class Imbalance

Because class 0 represented only **7.84%** of the dataset, imbalance-handling techniques were investigated.

## Experiment 1 — Balanced Sample Weights

Balanced sample weights were calculated using:

```python
compute_sample_weight(
    class_weight="balanced",
    y=y_train
)
```

The objective was to make mistakes on the minority class more important during training.

However, minority-class recall remained poor.

---

## Experiment 2 — SMOTE

SMOTE was then tested.

Before SMOTE:

```text
Class 0 → 627
Class 1 → 7373
```

After SMOTE:

```text
Class 0 → 7373
Class 1 → 7373
```

The test data was **not** resampled.

Despite balancing the training data, the minority-class performance remained extremely poor.

---

# 13. Model Evaluation

The following evaluation metrics were used:

* Accuracy
* Precision
* Recall
* F1-score
* Confusion matrix
* ROC-AUC
* PR-AUC

These metrics provide a much better picture than accuracy alone for an imbalanced classification problem.

---

# 14. SMOTE Results

The final SMOTE experiment produced:

```text
Accuracy: 0.922

ROC-AUC: 0.599
PR-AUC: 0.941
```

Confusion matrix:

```text
[[   1  156]
 [   0 1843]]
```

For class `0`:

```text
Precision: 1.00
Recall:    0.01
F1-score:  0.01
```

The model correctly detected only **1 out of 157 class-0 test examples**.

### Interpretation

Although the accuracy appears high, the model is essentially failing to detect the minority class.

This demonstrates why **accuracy alone is misleading** for this dataset.

The ROC-AUC was also only around **0.60**, indicating weak class-separation performance.

---

# 15. Key Findings

## Good aspects of the dataset

The dataset is useful for:

* Learning and demonstrating the full EDA workflow
* Data cleaning practice
* Handling numerical and categorical variables
* Visualization
* Feature encoding
* Train/test splitting
* Data leakage awareness
* Class-imbalance analysis
* Model evaluation
* Understanding precision, recall, F1-score, ROC-AUC and PR-AUC
* Experimenting with XGBoost and SMOTE

The dataset is also clean from a basic data-quality perspective:

* No missing values
* No duplicate rows
* Clear feature names
* A mixture of numerical and categorical features

---

# 16. Limitations of the Dataset

## 1. Severe Class Imbalance

The target distribution is:

```text
92.16% → class 1
7.84%  → class 0
```

This makes minority-class prediction difficult.

## 2. Weak Feature-Target Relationships

The numerical features show very weak individual correlations with the target.

## 3. Strong Feature Overlap

The exploratory analysis suggests that class 0 and class 1 observations often occupy similar feature ranges.

This makes it difficult for a classifier to find a reliable boundary.

## 4. Similar Categorical Distributions

Many categorical groups have target percentages close to the overall class distribution, suggesting limited separation from individual categorical features.

## 5. SMOTE Did Not Solve the Problem

Balancing the training set did not produce good minority-class recall.

This indicates that **class imbalance is not the only problem**.

The dataset also appears to have limited predictive separation.

---

# 17. Why More Complex Models May Not Solve the Problem

A more complex model, including a deep-learning model, can learn more complicated relationships.

However:

> **A more powerful algorithm cannot reliably create predictive information that is not present in the features.**

If class 0 and class 1 contain highly overlapping observations with very similar feature combinations, increasing model complexity may not provide a meaningful improvement.

Therefore, moving immediately to deep learning would not be the most justified next step.

The issue appears to be primarily **data quality, feature informativeness, target separability, and class imbalance**, rather than simply model complexity.

---

# 18. Final Conclusion

This project demonstrates an important data-science lesson:

> **A successful machine-learning project does not always end with a successful predictive model.**

The analysis shows that this dataset is useful for:

**EDA + preprocessing + feature analysis + imbalance handling + model evaluation**

but the available features provide insufficient evidence for reliable separation of the target classes.

The experiments with XGBoost, balanced sample weighting, and SMOTE did not produce satisfactory minority-class performance.

Therefore, under the current feature set and modeling experiments:

> **Reliable prediction of `Has_Mental_Health_Issue` is not well supported by the available data.**

Further improvement would require **better or more informative features, more reliable target labels, improved feature engineering, or better-quality data**, rather than simply increasing model complexity.

---

# 19. Key Learnings

### Data Understanding

* Understand the structure and meaning of features before modeling.
* Numerical dtype does not automatically mean a variable should be interpreted as continuous.
* Binary and discrete-count variables require appropriate visualizations.

### EDA

* Histograms show numerical distributions.
* Count plots are appropriate for categorical/binary values.
* Boxplots help understand spread and potential outliers.
* Correlation is useful but does not capture every possible relationship.
* Feature distributions should also be compared across target classes.

### Machine Learning

* Class imbalance can make accuracy misleading.
* Stratified splitting is important for imbalanced classification.
* Data leakage must be controlled.
* Preprocessing should be fitted only on training data.
* SMOTE should be applied only to training data.
* Recall and F1-score can be much more informative than accuracy for minority-class detection.

### Most Important Learning

> **Before searching for a more powerful algorithm, determine whether the data actually contains enough information to solve the problem.**

---

# 20. Project Structure

```text
Mental-Health/
│
├── data/
│   └── mental_health.csv
│
├── notebook/
│   └── mental_health.ipynb
│
├── src/
│   └── preprocess.py
│
├── train.py
├── README.md
├── requirements.txt
│
└── .gitignore
```

### File Description

| File                           | Purpose                          |
| ------------------------------ | -------------------------------- |
| `data/mental_health.csv`       | Dataset                          |
| `notebook/mental_health.ipynb` | EDA, analysis and visualizations |
| `src/preprocess.py`            | Data preprocessing pipeline      |
| `train.py`                     | Model training and evaluation    |
| `requirements.txt`             | Python dependencies              |
| `README.md`                    | Project documentation            |

---

# 21. Technologies Used

```text
Python
Pandas
NumPy
Matplotlib
Seaborn
Scikit-learn
XGBoost
Imbalanced-learn
Joblib
```

---

# 22. How to Run

Install the required dependencies:

```bash
pip install -r requirements.txt
```

Run the notebook for the EDA:

```text
notebook/mental_health.ipynb
```

Run model training:

```bash
python train.py
```

---

# 23. Final Project Takeaway

This project is not simply about finding the highest model accuracy.

It demonstrates the ability to:

**Inspect → Clean → Explore → Question → Prepare → Model → Evaluate → Interpret → Decide**

The final decision from the analysis is that the current dataset is **more suitable for understanding and demonstrating the data-science workflow than for building a reliable mental-health classification system**.
