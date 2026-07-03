from src.preprocessing import DataPreprocessor
from src.trainer import ModelTrainer
from src.evaluator import ModelEvaluator


def main():


    preprocessor = DataPreprocessor(
        "data/raw/data_B.csv"
    )

    X_train, X_test, y_train, y_test, preprocessing_pipeline = (
        preprocessor.run()
    )


    trainer = ModelTrainer(
        preprocessing_pipeline
    )

    trainer.run_all_models(
        X_train,
        X_test,
        y_train,
        y_test
    )


    evaluator = ModelEvaluator(
        trainer.best_model
    )

    metrics = evaluator.evaluate(
        X_test,
        y_test
    )

    evaluator.print_metrics(
        metrics
    )


if __name__ == "__main__":
    main()