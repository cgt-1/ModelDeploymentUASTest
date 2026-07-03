from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

import joblib

from src.preprocessing import DataPreprocessor


def main():

    preprocessor = DataPreprocessor(
        "data/raw/data_B.csv"
    )

    preprocessor.load_data()
    preprocessor.remove_unused_columns()
    preprocessor.standardize_missing_values()
    preprocessor.clean_numeric_columns()
    preprocessor.clean_categorical_columns()
    preprocessor.feature_engineering()
    preprocessor.validate_data()
    preprocessor.impute_missing_values()

    df = preprocessor.df


    streamlit_features = [
        "Age",
        "Occupation",
        "Annual_Income",
        "Monthly_Inhand_Salary",
        "Num_Bank_Accounts",
        "Num_Credit_Card",
        "Interest_Rate",
        "Delay_from_due_date",
        "Changed_Credit_Limit",
        "Num_Credit_Inquiries",
        "Credit_Mix",
        "Outstanding_Debt",
        "Credit_Utilization_Ratio",
        "Credit_History_Age",
        "Total_EMI_per_month",
        "Monthly_Balance",
        "Payment_of_Min_Amount",
        "Loan_Count",
        "Credit_Score"
    ]

    df_streamlit = df[
        streamlit_features
    ].copy()

    X = df_streamlit.drop(
        "Credit_Score",
        axis=1
    )

    y = df_streamlit["Credit_Score"].map({
        "Poor": 0,
        "Standard": 1,
        "Good": 2
    })

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )


    categorical_cols = (
        X_train
        .select_dtypes(include="object")
        .columns
        .tolist()
    )

    numerical_cols = (
        X_train
        .select_dtypes(exclude="object")
        .columns
        .tolist()
    )

    preprocessor_s = ColumnTransformer(

        transformers=[

            ("num",
                StandardScaler(),
                numerical_cols
            ),

            ("cat",
                OneHotEncoder(
                    handle_unknown="ignore"
                ),
                categorical_cols
            )

        ]

    )


    rf_streamlit = Pipeline([

        ("preprocessor",
            preprocessor_s
        ),

        ("model",
            RandomForestClassifier(
                n_estimators=100,
                random_state=42
            )
        )

    ])

    rf_streamlit.fit(
        X_train,
        y_train
    )

    y_pred = rf_streamlit.predict(
        X_test
    )

    print(
        "Accuracy:",
        accuracy_score(
            y_test,
            y_pred
        )
    )

    print(
        classification_report(
            y_test,
            y_pred
        )
    )

    print(
        confusion_matrix(
            y_test,
            y_pred
        )
    )

    print(
        "Training Accuracy:",
        rf_streamlit.score(
            X_train,
            y_train
        )
    )

    print(
        "Testing Accuracy:",
        rf_streamlit.score(
            X_test,
            y_test
        )
    )

    joblib.dump(
        rf_streamlit,
        "models/rf_streamlit.pkl"
    )

    print(
        "\nStreamlit model saved successfully!"
    )


if __name__ == "__main__":
    main()