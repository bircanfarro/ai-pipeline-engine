from abc import ABC, abstractmethod

class ProcessingStep(ABC):

    @abstractmethod
    def apply(self, text:str) -> str:
        pass
