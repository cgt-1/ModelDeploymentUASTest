import joblib
import pandas as pd


class CreditScorePredictor:

    def __init__(
        self,
        model_path="models/rf_streamlit.pkl"
    ):

        self.model = joblib.load(
            model_path
        )

    def predict(
        self,
        input_data
    ):

        if isinstance(input_data, dict):

            input_data = pd.DataFrame(
                [input_data]
            )

        prediction = self.model.predict(
            input_data
        )[0]

        label_mapping = {
            0: "Poor",
            1: "Standard",
            2: "Good"
        }

        return label_mapping[
            prediction
        ]