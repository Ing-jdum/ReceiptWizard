from abc import ABC, abstractmethod


class INerService(ABC):

    @abstractmethod
    def get_entities(self, input_string):
        pass
