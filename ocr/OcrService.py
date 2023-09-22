import os
from google.cloud import vision
from ocr.IOcrService import IOcrService
from ocr.google_ocr import prepare_image_local, VisionAI


class OcrService(IOcrService):

    def get_text(self, image_file_path):
        image = prepare_image_local(image_file_path)
        os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'resources/level-bond-399502-d6ce3cad0294.json'
        client = vision.ImageAnnotatorClient()
        va = VisionAI(client, image)
        texts = va.text_detection()

        return texts[0].description