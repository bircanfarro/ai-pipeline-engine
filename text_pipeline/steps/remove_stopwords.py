from text_pipeline.steps.base import ProcessingStep

class RemoveStopwords(ProcessingStep):

    def __init__(self, stopwords:list[str]) -> str:
        self.stopwords = [word.lower() for word in stopwords]

    def apply(self, text:str) -> str:
        return " ".join([word for word in text.split() if word.lower() not in self.stopwords])
