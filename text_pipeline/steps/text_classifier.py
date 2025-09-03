from steps.base import ProcessingStep
import joblib

class TextClassifier(ProcessingStep):

    def __init__(self, model_path: str = "sentiment_model.pkl"):
        self.model = joblib.load(model_path)

    def apply(self, text: str) -> str:
        return self.model.predict([text])[0]
