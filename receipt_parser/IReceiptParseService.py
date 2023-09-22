from abc import ABC, abstractmethod


class IReceiptParseService(ABC):
    @abstractmethod
    def parse_receipt(self, image_file_path):
        pass
