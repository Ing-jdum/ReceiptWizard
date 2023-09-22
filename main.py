from receipt_parser.ReceiptParseService import ReceiptParseService

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    image_path = 'resources/test'
    print(ReceiptParseService().parse_receipt(image_path))
