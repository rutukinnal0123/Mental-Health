# src/preprocess.py

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder


# ---------------------------------------------------------
# 1. Columns with a meaningful order
# ---------------------------------------------------------

ordinal_columns = [
    "Exercise_Per_Week",
    "Income_Level",
    "Alcohol_Frequency",
    "Diet_Quality",
    "Discuss_Mental_Health"
]


# Define the correct order for each ordinal column.
# The order matters because the encoder will convert
# categories into numbers based on this order.

ordinal_categories = [
    ["never", "1-2 times", "3-4 times", "5+ times"],
    ["low", "middle", "high"],
    ["never", "rarely", "weekly", "daily"],
    ["poor", "average", "good", "excellent"],
    ["never", "rarely", "sometimes", "yes easily"]
]


# ---------------------------------------------------------
# 2. Nominal categorical columns
# ---------------------------------------------------------

# These categories have NO natural order.
# Example:
# India < USA < Germany does not have any meaningful order.

nominal_columns = [
    "Gender",
    "Country",
    "Education",
    "Marital_Status",
    "Employment_Status",
    "Remote_Work",
    "Company_Mental_Health_Support",
    "Smoking"
]


# ---------------------------------------------------------
# 3. Create the preprocessing transformer
# ---------------------------------------------------------

preprocessor = ColumnTransformer(
    transformers=[

        # Ordinal encoding
        (
            "ordinal",
            OrdinalEncoder(
                categories=ordinal_categories,
                handle_unknown="use_encoded_value",
                unknown_value=-1
            ),
            ordinal_columns
        ),

        # One-hot encoding
        (
            "nominal",
            OneHotEncoder(
                handle_unknown="ignore",
                drop="first",
                sparse_output=False
            ),
            nominal_columns
        )
    ],

    # All other numerical/binary columns remain unchanged
    remainder="passthrough"
)