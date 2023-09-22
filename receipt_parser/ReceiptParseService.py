from ocr.OcrService import OcrService
from ner.NerService import NerService
from receipt_parser.IReceiptParseService import IReceiptParseService


class ReceiptParseService(IReceiptParseService):

    def parse_receipt(self, image_file_path):
        text = OcrService().get_text(image_file_path)
        return NerService().get_entities(text)
