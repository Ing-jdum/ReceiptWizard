import os
import openai


class GptClient:
    openai.api_key = os.environ.get('CHATGPT_API_KEY')

    _SYSTEM_PROMPT = ("You are a smart and intelligent Named Entity Recognition (NER) system. I will provide you the "
                      "definition of the entities you need to extract, the sentence from where your extract the "
                      "entities"
                      "and the output format with examples. But for reference, you will be extracting entities from "
                      "Dominican Republic receipts")

    _USER_PROMPT_1 = "Are you clear about your role?"

    _ASSISTANT_PROMPT_1 = ("Sure, I'm ready to help you with your NER task. Please provide me with the necessary "
                           "information to get started.")

    _GUIDELINES_PROMPT = (
        """
        Entity Definition:
            1. RNC: (Registro Nacional de Contribuyentes): The RNC is the tax identification number of a business or individual in the Dominican Republic. It's used for tax-related purposes and is usually printed prominently on receipts.
            2. NCF: is a unique identification code for each fiscal receipt or invoice. It helps in tracking and accounting for transactions. NCFs are typically alphanumeric and follow specific formats.
            3. FECHA: The date of the transaction, indicating when the goods or services were provided or purchased.
            4. TOTAL: The total amount due for the goods or services, often broken down into categories such as subtotal, ITBIS (Impuesto a la Transferencia de Bienes Industrializados y Servicios or Value Added Tax), and the final total. Usually the biggest currency format number.
            5. ITBIS (Impuesto a la Transferencia de Bienes Industrializados y Servicios): ITBIS is the Value Added Tax (VAT) in the Dominican Republic. It's a consumption tax applied to the value added at each stage of production or distribution. Normally it is 18% of the total.
            6. SUBTOTAL: The subtotal amount before taxes and other charges, a little less than the total.
            7. NOMBRE: The legal name of the business or individual providing the goods or services.
            8. DIRECCION: The physical address of the business or individual, which may include street, city, and contact information.
            9. PRODUCTOS: A detailed list of the goods or services provided, including their descriptions, quantities, unit prices, and total prices.
            10. METODO_PAGO: The payment method used for the transaction, such as cash, credit card, or check please use these values: EFECTIVO, TARJETA_CREDITO, CHEQUE.
        
        Output Format:
             {{'RNC': [list of entities present], 'NCF': [list of entities present], 'FECHA': Entity present, 'TOTAL': entity present, 'ITBIS': entity present, 'SUBTOTAL': entity present, 'NOMBRE': entity present, 'DIRECCION': entity present, 'PRODUCTOS':[list of entities present], 'METODO_PAGO': entity present}}
        
        Examples:
    
            1. Sentence: 20:17:00 COL MAPP POS AUT 30 DE MAYO KM & 12 SANTO DOMINGO , DO CERVECERIA NACIONAL DOMINICANA AZUL ID Comercio : 000000008205409 ID Term : 01141475 Comercio - Sur H : Pagos Rapidos Venta XXXXXXXXXXXX9559 Mastory and Metodo Entrada . Sin Contacto Procesado . En Linea 01/10/22 Etiqueta : DEBIT MASTERCARD AID : A0000000041010 AROC : 2E5D89FE3E220935 Tran # 1 : 0000000363 Lote # 1 : 000054 Monto : ITBIS : Total : DOP DOP DEBIT MASTERCARD 2 Aprobacion # : 060857 440.68 79.32 520.00 DOP 39243440212 FIRMA NO REQUERIDA Copia Cliente 20:17:00 
            Output: 
            {{
                "RNC": ["ORG: CERVECERIA NACIONAL DOMINICANA"],
                "NCF": [],
                "FECHA": "01/10/22",
                "TOTAL": "520.00 DOP",
                "SUBTOTAL": "440.68",
                "ITBIS": "79.32 DOP",
                "NOMBRE": "",
                "DIRECCION": "AUT 30 DE MAYO KM & 12 SANTO DOMINGO",
                "PRODUCTOS": []
            }}
            
            2. Sentence: V: 3.00 Taurus GASTRONOMIA MEXICANA , S.R.L. RNC 130056358 RES DGII : 23-2009 DEL 06 / ABRIL / 2009 COMPROBANTE AUTORIZADO POR DGII 17/09/2023 17:41:23 NIF : 1070260000224621 NCF : 00000000B0100022641 RNC : 132629086 ROSOINCO SRL FACTURA PARA CREDITO FISCAL DESCRIPCION 1.00 x 495.00 CHIMICHANGAAO - MIXTO 1.00 x 140.00 Dosis 11oz . JUGO NARANJA NATURAL / 11OZ . 1.00 x 100.00 B.13.5oz . REFRESCO C.COLA / 12OZ . 1.00 x 490.00 BURRO ENCENDIDO Efectivo TARJETA CREDITO FACTOR DE CONV . 1568.00 CAMBIO Subtotal ITBIS 18 % TOTAL ITBIS % LEY TOTAL Caja : 1 / Vendedor : Rosita Serie : T001-284569 / Uds .: 4 / Sala - Mesa : 8-3 NIF : 1070260000224621 P4YF002277 II 107026 VALOR 495.00 140.00 100.00 490.00 1,225.00 220.50 220.50 122.50 1,568.00 1,568.00 0.00 V : 3.00 Taurus 
            Output: 
            {{
                "RNC": ["130056358", "132629086"],
                "NCF": ["00000000B0100022641"],
                "FECHA": "17/09/2023 17:41:23",
                "TOTAL": "1,568.00",
                "SUBTOTAL": "1,225.00",
                "ITBIS": "220.50",
                "NOMBRE": "GASTRONOMIA MEXICANA, S.R.L.",
                "DIRECCION": "",
                "PRODUCTOS": [
                    {{"Cantidad": "1.00", "Precio": "495.00", "Descripcion": "CHIMICHANGAAO - MIXTO"}},
                    {{"Cantidad": "1.00", "Precio": "140.00", "Descripcion": "Dosis 11oz . JUGO NARANJA NATURAL / 11OZ ." }},
                    {{"Cantidad": "1.00", "Precio": "100.00", "Descripcion": "B.13.5oz . REFRESCO C.COLA / 12OZ ." }},
                    {{ "Cantidad": "1.00", "Precio": "490.00", "Descripcion": "BURRO ENCENDIDO"}}
                ]
            }}
            3. Sentence: {}
            Output: 
        """
    )

    def openai_chat_completion_response(self, final_prompt):
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": self._SYSTEM_PROMPT},
                {"role": "user", "content": self._USER_PROMPT_1},
                {"role": "assistant", "content": self._ASSISTANT_PROMPT_1},
                {"role": "user", "content": final_prompt}
            ]
        )

        return response['choices'][0]['message']['content'].strip(" \n")

    def get_entities(self, input_string):
        command = self._GUIDELINES_PROMPT.format(input_string)
        return self.openai_chat_completion_response(command)

# my_sentence = ("""ESTACION SHELL TENARES
# C/DUARTE ESQ. CRUZ PORTES., TANARES, SALCEDO, RE
# P. DOM.
# RNC 131214991
# TEL 809-587-8452
# C
# LL FECHA: 18/09/2023
# TO FACTURA DE CONTADO
# VALIDO PARA CREDITO FISCAL
# NCF B0100009260 FACT# 20011271
# NCF Valido hasta : 31/12/2023
# ROSOINCO
# RNC: 132629086
# COMBUSTIBLE
# LUBRICANTE
# HORA 121745
# TOTAL
# 6,000.00
# 0.00
# 6,000.00
# NICOLELL""")
#
# GUIDELINES_PROMPT = GUIDELINES_PROMPT.format(my_sentence)
# ners = openai_chat_completion_response(GUIDELINES_PROMPT)
# print(ners)
