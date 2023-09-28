import os
from google.cloud import vision
from ocr.IOcrService import IOcrService
from ocr.google_ocr import VisionAI, prepare_image_local, draw_boundary, draw_boundary_normalized


class OcrService(IOcrService):

    def get_text(self, image_url):
        image = prepare_image_local(image_url)
        os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'resources/level-bond-399502-d6ce3cad0294.json'
        client = vision.ImageAnnotatorClient()
        va = VisionAI(client, image)
        texts = va.text_detection()
        result = self.to_format(texts)
        print(result)
        return result

    def calc_euclidian_distance(self, point):
        x, y = point
        HEIGHT_WEIGHT = 10
        distance = x ** 2 + HEIGHT_WEIGHT*y ** 2
        return distance


    def to_format(self, texts):
        dict = {'values': []}
        for text in texts[1:len(texts)]:
            xys = [(vertex.x, vertex.y) for vertex in text.bounding_poly]
            dict['values'].append({'description': text.description, 'polygon': xys})

        return dict

    def sort_text(self, texts):
        dict = {}
        for text in texts[1:len(texts)]:
            bottom_right_vertex = (text.bounding_poly[0].x, text.bounding_poly[3].y)
            dict[text.description] = self.calc_euclidian_distance(bottom_right_vertex)

        sorted_dict = {k: v for k, v in sorted(dict.items(), key=lambda item: item[1])}
        return ' '.join(sorted_dict.keys())
