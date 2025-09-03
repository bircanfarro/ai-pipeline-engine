from text_pipeline.steps.base import ProcessingStep

class Lowercase(ProcessingStep):

    def apply(self, text:str) -> str:
        return text.lower()
