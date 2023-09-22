from receipt_parser.ReceiptParseService import ReceiptParseService

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    image_url = 'https://templates.invoicehome.com/modelo-factura-es-puro-750px.png'
    print(ReceiptParseService().parse_receipt(image_url))
