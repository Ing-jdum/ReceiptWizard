from ner.GptClient import GptClient
from ner.INerService import INerService


class NerService(INerService):
    def get_entities(self, input_string):
        return GptClient().get_entities(input_string)