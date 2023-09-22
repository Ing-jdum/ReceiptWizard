from abc import ABC, abstractmethod


class IOcrService(ABC):
    @abstractmethod
    def get_text(self, image_file_path):
        pass
