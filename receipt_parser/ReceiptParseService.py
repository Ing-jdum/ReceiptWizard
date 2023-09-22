from ocr.OcrService import OcrService
from ner.NerService import NerService
from receipt_parser.IReceiptParseService import IReceiptParseService


class ReceiptParseService(IReceiptParseService):

    def parse_receipt(self, image_url):
        text = OcrService().get_text(image_url)
        return NerService().get_entities(text)
