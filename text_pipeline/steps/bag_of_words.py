from steps.base import ProcessingStep
from collections import Counter

class BagOfWordVectorizer(ProcessingStep):

    def apply(self, text: str) -> dict[str, int]:
        words = text.split()
        return dict(Counter(words))
