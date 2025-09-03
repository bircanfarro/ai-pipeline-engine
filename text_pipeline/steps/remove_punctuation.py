import re
from steps.base import ProcessingStep

class RemovePunctuation(ProcessingStep):

    def apply(self, text:str) -> str:
        return re.sub(r"[^\w\s]", "", text)
