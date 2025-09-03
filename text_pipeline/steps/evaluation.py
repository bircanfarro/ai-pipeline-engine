from text_pipeline.steps.base import ProcessingStep
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

class EvaluationStep(ProcessingStep):
    """Evaluates prediction performance against ground truth."""

    def __init__(self, true_labels: list[str]):
        self.true_labels = true_labels

    def apply(self, predicted_labels: list[str]) -> dict[str, float]:
        return {
            "accuracy": accuracy_score(self.true_labels, predicted_labels),
            "precision": precision_score(self.true_labels, predicted_labels, pos_label="positive"),
            "recall": recall_score(self.true_labels, predicted_labels, pos_label="positive"),
            "f1": f1_score(self.true_labels, predicted_labels, pos_label="positive")
        }
