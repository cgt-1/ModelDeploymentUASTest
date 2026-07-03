import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer


class DataPreprocessor:

    def __init__(self, data_path):
        self.data_path = data_path
        self.df = None
        self.loan_types = None

    def load_data(self):
        self.df = pd.read_csv(self.data_path)
        return self.df

    def remove_unused_columns(self):

        self.df = self.df.drop(columns=[
            "Unnamed: 0",
            "ID",
            "Customer_ID",
            "Name",
            "SSN"
        ])

    def standardize_missing_values(self):

        self.df = self.df.replace(
            [
                "",
                " ",
                "  ",
                "-",
                "--",
                "N/A",
                "NA"
            ],
            np.nan
        )

        self.df = self.df.replace(
            r"^_+$",
            np.nan,
            regex=True
        )

    def clean_numeric_columns(self):

        numeric_cols = [
            "Age",
            "Annual_Income",
            "Monthly_Inhand_Salary",
            "Num_Bank_Accounts",
            "Num_Credit_Card",
            "Interest_Rate",
            "Num_of_Loan",
            "Delay_from_due_date",
            "Num_of_Delayed_Payment",
            "Changed_Credit_Limit",
            "Num_Credit_Inquiries",
            "Outstanding_Debt",
            "Credit_Utilization_Ratio",
            "Total_EMI_per_month",
            "Amount_invested_monthly",
            "Monthly_Balance"
        ]

        for col in numeric_cols:

            self.df[col] = (
                self.df[col]
                .astype(str)
                .str.replace(
                    r"[^0-9.-]",
                    "",
                    regex=True
                )
            )

            self.df[col] = pd.to_numeric(
                self.df[col],
                errors="coerce"
            )

    def clean_categorical_columns(self):

        self.df["Payment_Behaviour"] = (
            self.df["Payment_Behaviour"]
            .replace("!@9#%8", np.nan)
        )

        self.df["Payment_of_Min_Amount"] = (
            self.df["Payment_of_Min_Amount"]
            .replace("NM", np.nan)
        )

    def feature_engineering(self):

        def convert_history_age(x):

            if pd.isna(x):
                return np.nan

            parts = x.split()

            years = int(parts[0])
            months = int(parts[3])

            return years * 12 + months

        self.df["Credit_History_Age"] = (
            self.df["Credit_History_Age"]
            .apply(convert_history_age)
        )

        self.df["Type_of_Loan"] = (
            self.df["Type_of_Loan"]
            .str.replace(
                " and ",
                ", ",
                regex=False
            )
        )

        all_loans = set()

        for loans in self.df["Type_of_Loan"].dropna():

            for loan in loans.split(","):
                all_loans.add(
                    loan.strip()
                )

        self.loan_types = sorted(all_loans)

        if "Not Specified" in self.loan_types:
            self.loan_types.remove("Not Specified")
        
        for loan in self.loan_types:

            col_name = (
                loan.replace(" ", "_")
                    .replace("-", "_")
            )

            self.df[col_name] = (
                self.df["Type_of_Loan"]
                .fillna("")
                .str.contains(
                    loan,
                    regex=False
                )
                .astype(int)
            )

        def count_loans(x):

            if pd.isna(x):
                return np.nan

            count = 0

            for loan in self.loan_types:
                if loan in x:
                    count += 1

            return count

        self.df["Loan_Count"] = (
            self.df["Type_of_Loan"]
            .apply(count_loans)
        )

        self.df = self.df.drop(columns=[
            "Type_of_Loan",
            "Num_of_Loan"
        ])

        if "" in self.df.columns:
            self.df = self.df.drop(columns=[""])

    def validate_data(self):

        validation_rules = {

            "Age": (18, 100),

            "Num_Bank_Accounts": (0, 20),

            "Num_Credit_Card": (0, 20),

            "Interest_Rate": (0, 100),

            "Delay_from_due_date": (0, None),

            "Num_of_Delayed_Payment": (0, 100),

            "Num_Credit_Inquiries": (0, 50),

            "Monthly_Balance": (0, None)

        }

        for col, (min_val, max_val) in validation_rules.items():

            if min_val is not None:
                self.df.loc[
                    self.df[col] < min_val,
                    col
                ] = np.nan

            if max_val is not None:
                self.df.loc[
                    self.df[col] > max_val,
                    col
                ] = np.nan

    def impute_missing_values(self):

        num_cols = self.df.select_dtypes(
            include=["int64", "float64"]
        ).columns

        for col in num_cols:

            self.df[col].fillna(
                self.df[col].median(),
                inplace=True
            )

        cat_cols = self.df.select_dtypes(
            include="object"
        ).columns

        for col in cat_cols:

            self.df[col].fillna(
                self.df[col].mode()[0],
                inplace=True
            )

    def save_cleaned_data(
        self,
        output_path="data/processed/cleaned_data.csv"
    ):

        self.df.to_csv(
            output_path,
            index=False
        )

    def prepare_training_data(self):

        self.df = self.df.drop(columns=["Month"])

        X = self.df.drop("Credit_Score", axis=1)

        y = self.df["Credit_Score"].map({
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

        return (
            X_train,
            X_test,
            y_train,
            y_test
        )

    def build_preprocessor(self, X_train):

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

        preprocessor = ColumnTransformer(
            transformers=[
                (
                    "num",
                    StandardScaler(),
                    numerical_cols
                ),
                (
                    "cat",
                    OneHotEncoder(handle_unknown="ignore"),
                    categorical_cols
                )
            ]
        )

        return preprocessor

    def run(self):

        self.load_data()

        self.remove_unused_columns()

        self.standardize_missing_values()

        self.clean_numeric_columns()

        self.clean_categorical_columns()

        self.feature_engineering()

        self.validate_data()

        self.impute_missing_values()

        self.save_cleaned_data()

        X_train, X_test, y_train, y_test = (
            self.prepare_training_data()
        )

        preprocessor = self.build_preprocessor(
            X_train
        )

        return (
            X_train,
            X_test,
            y_train,
            y_test,
            preprocessor
        )
    