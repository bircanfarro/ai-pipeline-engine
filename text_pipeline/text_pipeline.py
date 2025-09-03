from text_pipeline.steps.base import ProcessingStep

class TextPipeline:
    """Runs a series of processing steps on text."""

    def __init__(self, steps: list[ProcessingStep]):
        self.steps = steps

    def process(self, text: str) -> str:
        for step in self.steps:
            text = step.apply(text)
        return text
