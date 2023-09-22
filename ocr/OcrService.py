import os
from google.cloud import vision
from ocr.IOcrService import IOcrService
from ocr.google_ocr import prepare_image_web, VisionAI


class OcrService(IOcrService):

    def get_text(self, image_url):
        image = prepare_image_web(image_url)
        os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'resources/level-bond-399502-d6ce3cad0294.json'
        client = vision.ImageAnnotatorClient()
        va = VisionAI(client, image)
        texts = va.text_detection()

        return texts[0].description
