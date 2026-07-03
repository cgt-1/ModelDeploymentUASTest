from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)


class ModelEvaluator:

    def __init__(self, model):

        self.model = model

    def evaluate(
        self,
        X_test,
        y_test
    ):

        y_pred = self.model.predict(
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

        print("\n===== Evaluation =====")

        print(
            classification_report(
                y_test,
                y_pred
            )
        )

        cm = confusion_matrix(
            y_test,
            y_pred
            )
        print(cm)

        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "confusion_matrix": cm
        }
    
    def print_metrics(
        self,
        metrics
    ):

        print("\nModel Performance")

        print(
            f"Accuracy : {metrics['accuracy']:.4f}"
        )

        print(
            f"Precision: {metrics['precision']:.4f}"
        )

        print(
            f"Recall   : {metrics['recall']:.4f}"
        )

        print(
            f"F1 Score : {metrics['f1_score']:.4f}"
        )