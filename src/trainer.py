import os
import joblib
import mlflow
import mlflow.sklearn

from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from xgboost import XGBClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)


class ModelTrainer:

    def __init__(self, preprocessor):

        self.preprocessor = preprocessor

        self.best_model = None
        self.best_accuracy = 0

    def train_model(
        self,
        model_name,
        model,
        X_train,
        X_test,
        y_train,
        y_test
    ):

        pipeline = Pipeline([

            ("preprocessor", self.preprocessor),

            ("model", model)

        ])

        pipeline.fit(
            X_train,
            y_train
        )

        y_pred = pipeline.predict(
            X_test
        )

        accuracy = accuracy_score(
            y_test,
            y_pred
        )

        precision = precision_score(
            y_test,
            y_pred,
            average="weighted"
        )

        recall = recall_score(
            y_test,
            y_pred,
            average="weighted"
        )

        f1 = f1_score(
            y_test,
            y_pred,
            average="weighted"
        )

        with mlflow.start_run(run_name=model_name):

            mlflow.log_param(
                "model",
                model_name
            )

            mlflow.log_metric(
                "accuracy",
                accuracy
            )

            mlflow.log_metric(
                "precision",
                precision
            )

            mlflow.log_metric(
                "recall",
                recall
            )

            mlflow.log_metric(
                "f1_score",
                f1
            )

            print(f"\n===== {model_name} =====")

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

            if accuracy > self.best_accuracy:

                self.best_accuracy = accuracy

                self.best_model = pipeline

                mlflow.sklearn.log_model(
                    pipeline,
                    "best_model"
                )

        return pipeline
    
    def save_model(
        self,
        model,
        filename
    ):

        os.makedirs(
            "models",
            exist_ok=True
        )

        joblib.dump(
            model,
            os.path.join(
                "models",
                filename
            )
        )

    def save_best_model(self):

        self.save_model(
            self.best_model,
            "rf_best.pkl"
        )

        print(
            f"\nBest Model Accuracy: {self.best_accuracy:.4f}"
        )

    def train_streamlit_model(
        self,
        X_train,
        X_test,
        y_train,
        y_test
    ):

        streamlit_pipeline = Pipeline([

            ("preprocessor", self.preprocessor),

            ("model", RandomForestClassifier(
                n_estimators=100,
                random_state=42
            ))

        ])

        streamlit_pipeline.fit(
            X_train,
            y_train
        )

        y_pred = streamlit_pipeline.predict(
            X_test
        )

        accuracy = accuracy_score(
            y_test,
            y_pred
        )

        print("\nStreamlit Random Forest")

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

        self.save_model(
            streamlit_pipeline,
            "rf_streamlit.pkl"
        )

        return streamlit_pipeline
    
    def run_all_models(
        self,
        X_train,
        X_test,
        y_train,
        y_test
    ):

        self.train_model(
            "Logistic Regression",
            LogisticRegression(
                max_iter=1000
            ),
            X_train,
            X_test,
            y_train,
            y_test
        )

        self.train_model(
            "Decision Tree",
            DecisionTreeClassifier(
                random_state=42
            ),
            X_train,
            X_test,
            y_train,
            y_test
        )

        self.train_model(
            "Random Forest",
            RandomForestClassifier(
                n_estimators=100,
                random_state=42
            ),
            X_train,
            X_test,
            y_train,
            y_test
        )

        self.train_model(
            "XGBoost",
            XGBClassifier(
                objective="multi:softmax",
                num_class=3,
                eval_metric="mlogloss",
                random_state=42
            ),
            X_train,
            X_test,
            y_train,
            y_test
        )

        self.save_best_model()