import numpy as np
import math
import cv2
from sklearn.cluster import DBSCAN
import copy

import ocr

RESPONSE = {'values': [{'description': 'SIRENA', 'polygon': [(390, 105), (439, 113), (437, 130), (387, 122)]}, {'description': 'GALERIA', 'polygon': [(449, 115), (506, 125), (503, 141), (446, 132)]}, {'description': '360', 'polygon': [(514, 126), (539, 130), (536, 146), (511, 142)]}, {'description': 'RNC', 'polygon': [(311, 114), (335, 118), (332, 131), (309, 127)]}, {'description': '101796822', 'polygon': [(346, 120), (420, 133), (418, 145), (344, 133)]}, {'description': 'GAL.360', 'polygon': [(431, 135), (487, 145), (485, 157), (429, 148)]}, {'description': '809-563-4111', 'polygon': [(497, 146), (591, 162), (588, 175), (495, 159)]}, {'description': 'GRUPO', 'polygon': [(379, 141), (420, 149), (418, 162), (377, 154)]}, {'description': 'RAMOS', 'polygon': [(430, 151), (470, 159), (468, 170), (428, 163)]}, {'description': 'S.A.', 'polygon': [(480, 160), (509, 165), (506, 177), (478, 172)]}, {'description': 'RNC', 'polygon': [(308, 148), (332, 152), (330, 164), (306, 160)]}, {'description': ':', 'polygon': [(336, 153), (340, 154), (338, 164), (334, 164)]}, {'description': '101796822', 'polygon': [(353, 155), (426, 166), (424, 178), (351, 167)]}, {'description': '21/09/23', 'polygon': [(307, 164), (374, 176), (372, 188), (305, 176)]}, {'description': '09:46:58', 'polygon': [(384, 178), (450, 190), (448, 202), (382, 190)]}, {'description': 'e', 'polygon': [(307, 182), (313, 183), (311, 194), (305, 193)]}, {'description': '-', 'polygon': [(316, 183), (322, 184), (320, 195), (314, 194)]}, {'description': 'NCF', 'polygon': [(324, 184), (347, 188), (344, 200), (322, 196)]}, {'description': ':', 'polygon': [(351, 190), (354, 191), (352, 201), (349, 201)]}, {'description': 'E320004795244', 'polygon': [(358, 190), (466, 210), (464, 222), (356, 202)]}, {'description': 'FACTURA', 'polygon': [(339, 222), (397, 233), (395, 245), (337, 234)]}, {'description': 'DE', 'polygon': [(407, 235), (422, 238), (419, 250), (405, 247)]}, {'description': 'CONSUMO', 'polygon': [(432, 240), (488, 251), (486, 263), (430, 252)]}, {'description': 'ELECTRONICA', 'polygon': [(498, 253), (584, 270), (582, 281), (496, 265)]}, {'description': 'DESCRIPCION', 'polygon': [(301, 253), (395, 268), (393, 280), (299, 266)]}, {'description': 'ABARROTES', 'polygon': [(299, 294), (374, 303), (373, 313), (298, 304)]}, {'description': '770241800484', 'polygon': [(298, 310), (401, 323), (400, 333), (297, 321)]}, {'description': 'MASTICAB', 'polygon': [(296, 327), (364, 335), (363, 348), (294, 340)]}, {'description': 'FRE', 'polygon': [(376, 337), (400, 340), (398, 352), (375, 349)]}, {'description': '841037603778', 'polygon': [(295, 345), (400, 358), (399, 372), (293, 359)]}, {'description': 'GALLETA', 'polygon': [(294, 363), (353, 371), (351, 385), (292, 377)]}, {'description': 'CHOC', 'polygon': [(365, 373), (399, 377), (397, 390), (363, 386)]}, {'description': 'PROD', 'polygon': [(292, 383), (326, 387), (324, 401), (290, 397)]}, {'description': '.', 'polygon': [(330, 387), (334, 387), (332, 401), (328, 401)]}, {'description': 'FRESCOS', 'polygon': [(346, 389), (406, 396), (404, 411), (344, 404)]}, {'description': '000000011091', 'polygon': [(291, 404), (394, 416), (393, 428), (290, 416)]}, {'description': 'WRAP', 'polygon': [(289, 423), (323, 427), (321, 439), (288, 435)]}, {'description': 'JAM', 'polygon': [(335, 428), (358, 431), (356, 443), (334, 440)]}, {'description': 'QUE', 'polygon': [(371, 432), (395, 435), (393, 447), (370, 444)]}, {'description': '074675028000', 'polygon': [(288, 441), (394, 453), (393, 465), (287, 453)]}, {'description': 'LUMIJOR', 'polygon': [(288, 459), (347, 466), (345, 478), (287, 471)]}, {'description': 'PAN', 'polygon': [(360, 468), (385, 471), (384, 482), (359, 479)]}, {'description': 'I', 'polygon': [(395, 472), (402, 473), (401, 484), (394, 483)]}, {'description': 'TOTAL', 'polygon': [(285, 499), (329, 505), (327, 518), (283, 513)]}, {'description': 'A', 'polygon': [(340, 506), (347, 507), (345, 520), (338, 519)]}, {'description': 'PAGAR', 'polygon': [(358, 509), (401, 514), (399, 527), (356, 522)]}, {'description': 'VISA', 'polygon': [(284, 519), (319, 523), (317, 535), (283, 531)]}, {'description': 'ITBIS', 'polygon': [(465, 281), (506, 289), (504, 301), (463, 293)]}, {'description': '0.00', 'polygon': [(470, 353), (503, 358), (501, 369), (468, 364)]}, {'description': '19.68', 'polygon': [(460, 387), (500, 394), (498, 405), (458, 398)]}, {'description': '21.36', 'polygon': [(456, 443), (497, 453), (494, 466), (453, 456)]}, {'description': 'Aprobacion', 'polygon': [(239, 1040), (333, 1043), (333, 1057), (239, 1054)]}, {'description': ':', 'polygon': [(337, 1043), (340, 1043), (340, 1056), (337, 1056)]}, {'description': '310013', 'polygon': [(354, 1043), (410, 1045), (410, 1059), (354, 1057)]}, {'description': 'Monto', 'polygon': [(236, 1062), (283, 1063), (283, 1078), (236, 1077)]}, {'description': 'RD', 'polygon': [(296, 1064), (313, 1064), (313, 1078), (296, 1078)]}, {'description': '$', 'polygon': [(316, 1065), (323, 1065), (323, 1079), (316, 1079)]}, {'description': ':', 'polygon': [(326, 1065), (330, 1065), (330, 1079), (326, 1079)]}, {'description': '619.00', 'polygon': [(343, 1065), (399, 1067), (399, 1082), (343, 1080)]}, {'description': 'ITBIS', 'polygon': [(235, 1086), (282, 1087), (282, 1101), (235, 1100)]}, {'description': 'RDS', 'polygon': [(294, 1087), (320, 1087), (320, 1101), (294, 1101)]}, {'description': ':', 'polygon': [(325, 1088), (329, 1088), (329, 1101), (325, 1101)]}, {'description': '41.04', 'polygon': [(342, 1088), (389, 1089), (389, 1103), (342, 1102)]}, {'description': '0.00', 'polygon': [(463, 486), (494, 489), (493, 500), (462, 497)]}, {'description': '41.04', 'polygon': [(452, 523), (495, 530), (493, 543), (450, 536)]}, {'description': 'VALOR', 'polygon': [(563, 300), (601, 305), (599, 316), (561, 311)]}, {'description': '205.00', 'polygon': [(552, 366), (600, 375), (598, 386), (550, 377)]}, {'description': 'E', 'polygon': [(607, 377), (614, 378), (612, 388), (605, 387)]}, {'description': '129.00', 'polygon': [(549, 405), (595, 413), (594, 423), (547, 415)]}, {'description': 'FECHA', 'polygon': [(244, 974), (290, 976), (289, 989), (243, 987)]}, {'description': ':', 'polygon': [(294, 977), (299, 977), (298, 989), (293, 989)]}, {'description': '21/09/2023', 'polygon': [(311, 977), (402, 981), (401, 994), (310, 990)]}, {'description': 'Modo', 'polygon': [(241, 996), (278, 997), (278, 1010), (241, 1009)]}, {'description': 'de', 'polygon': [(291, 998), (307, 999), (307, 1012), (291, 1011)]}, {'description': 'entrada', 'polygon': [(320, 999), (382, 1001), (381, 1014), (320, 1012)]}, {'description': ':', 'polygon': [(387, 1002), (390, 1002), (390, 1014), (387, 1014)]}, {'description': 'CHIP', 'polygon': [(404, 1002), (439, 1003), (439, 1016), (404, 1015)]}, {'description': 'No.', 'polygon': [(239, 1018), (265, 1019), (265, 1033), (239, 1032)]}, {'description': 'tarjeta', 'polygon': [(280, 1019), (344, 1021), (344, 1035), (280, 1033)]}, {'description': ':', 'polygon': [(348, 1021), (351, 1021), (351, 1034), (348, 1034)]}, {'description': '459413', 'polygon': [(365, 1021), (419, 1022), (419, 1036), (365, 1035)]}, {'description': '******', 'polygon': [(421, 1023), (474, 1024), (474, 1038), (421, 1037)]}, {'description': '2743', 'polygon': [(476, 1025), (512, 1026), (512, 1039), (476, 1038)]}, {'description': '140.00', 'polygon': [(547, 462), (592, 469), (591, 480), (545, 473)]}, {'description': '145.00', 'polygon': [(544, 500), (589, 506), (587, 517), (543, 511)]}, {'description': 'E', 'polygon': [(601, 508), (607, 509), (606, 519), (600, 518)]}, {'description': 'ASHLEY', 'polygon': [(282, 558), (335, 565), (334, 579), (280, 572)]}, {'description': 'GISSEL', 'polygon': [(346, 567), (394, 574), (392, 587), (344, 581)]}, {'description': 'SOTO', 'polygon': [(405, 575), (437, 579), (435, 593), (403, 589)]}, {'description': 'PUJOLS', 'polygon': [(448, 581), (495, 587), (493, 601), (446, 595)]}, {'description': '250004449488', 'polygon': [(506, 589), (605, 603), (603, 617), (504, 603)]}, {'description': 'NUMERO', 'polygon': [(317, 583), (369, 590), (367, 604), (315, 597)]}, {'description': 'ARTICULOS', 'polygon': [(379, 592), (452, 602), (451, 615), (377, 605)]}, {'description': 'VENDIDOS', 'polygon': [(463, 603), (529, 612), (527, 626), (461, 617)]}, {'description': '=', 'polygon': [(539, 614), (545, 615), (543, 628), (537, 627)]}, {'description': '4', 'polygon': [(553, 616), (560, 617), (558, 630), (551, 629)]}, {'description': '21/09/23', 'polygon': [(307, 602), (376, 611), (375, 625), (305, 616)]}, {'description': '09:46', 'polygon': [(386, 613), (425, 618), (423, 631), (384, 626)]}, {'description': '6019', 'polygon': [(437, 620), (468, 624), (466, 637), (435, 633)]}, {'description': '30', 'polygon': [(479, 625), (494, 627), (492, 640), (477, 638)]}, {'description': '0031', 'polygon': [(505, 629), (535, 633), (533, 646), (503, 642)]}, {'description': '7188', 'polygon': [(544, 634), (576, 638), (574, 651), (542, 647)]}, {'description': 'ESTA', 'polygon': [(277, 620), (312, 624), (310, 637), (275, 633)]}, {'description': 'COMPRA', 'polygon': [(324, 625), (375, 631), (373, 645), (322, 639)]}, {'description': 'LE', 'polygon': [(385, 633), (400, 635), (398, 648), (383, 646)]}, {'description': 'SUMA', 'polygon': [(410, 636), (442, 640), (440, 653), (408, 649)]}, {'description': 'A', 'polygon': [(453, 641), (459, 642), (457, 655), (451, 654)]}, {'description': 'SU', 'polygon': [(470, 643), (485, 645), (483, 658), (468, 656)]}, {'description': 'BALANCE', 'polygon': [(495, 646), (552, 653), (550, 666), (493, 659)]}, {'description': 'PUNTOS', 'polygon': [(409, 657), (458, 664), (456, 675), (407, 668)]}, {'description': '.', 'polygon': [(461, 665), (464, 665), (463, 675), (460, 675)]}, {'description': 'Le', 'polygon': [(276, 660), (291, 662), (289, 677), (274, 675)]}, {'description': 'Atendio', 'polygon': [(304, 663), (362, 669), (360, 684), (302, 678)]}, {'description': ':', 'polygon': [(375, 671), (379, 671), (377, 685), (373, 685)]}, {'description': 'NICAURY', 'polygon': [(391, 673), (450, 679), (448, 693), (389, 687)]}, {'description': 'A.', 'polygon': [(460, 680), (473, 681), (471, 696), (458, 695)]}, {'description': 'SANTANA', 'polygon': [(486, 683), (542, 689), (540, 703), (484, 697)]}, {'description': 'J', 'polygon': [(551, 690), (559, 691), (557, 705), (549, 704)]}, {'description': '619.00', 'polygon': [(541, 540), (588, 546), (586, 556), (540, 550)]}, {'description': '619.00', 'polygon': [(541, 556), (588, 563), (586, 575), (539, 568)]}, {'description': '23092160190300071880031', 'polygon': [(350, 748), (520, 762), (519, 776), (349, 762)]}, {'description': 'VALORAMOS', 'polygon': [(327, 767), (403, 773), (402, 787), (326, 781)]}, {'description': 'SU', 'polygon': [(414, 774), (429, 775), (428, 789), (413, 788)]}, {'description': 'PREFERENCIA', 'polygon': [(441, 776), (532, 783), (531, 797), (440, 790)]}, {'description': 'ASHLEY', 'polygon': [(317, 786), (367, 790), (366, 804), (316, 800)]}, {'description': 'GISSEL', 'polygon': [(379, 791), (428, 795), (427, 809), (378, 805)]}, {'description': 'SOTO', 'polygon': [(440, 796), (473, 799), (472, 811), (439, 809)]}, {'description': 'PUJOLS', 'polygon': [(484, 799), (533, 803), (532, 817), (483, 813)]}, {'description': 'CONSULTE', 'polygon': [(268, 802), (339, 808), (338, 821), (267, 816)]}, {'description': 'NUESTRA', 'polygon': [(351, 809), (410, 814), (409, 827), (350, 822)]}, {'description': 'POLITICA', 'polygon': [(422, 814), (490, 819), (489, 833), (421, 828)]}, {'description': 'DE', 'polygon': [(501, 820), (516, 821), (515, 835), (500, 834)]}, {'description': 'CAMBIOS', 'polygon': [(526, 822), (581, 826), (580, 840), (525, 836)]}, {'description': 'Y', 'polygon': [(256, 824), (263, 825), (262, 837), (255, 837)]}, {'description': 'DEVOLUCIONES', 'polygon': [(276, 825), (381, 833), (380, 846), (275, 838)]}, {'description': 'EN', 'polygon': [(393, 833), (408, 834), (407, 847), (392, 846)]}, {'description': 'SERVICIO', 'polygon': [(420, 835), (489, 840), (488, 854), (419, 849)]}, {'description': 'AL', 'polygon': [(501, 841), (515, 842), (514, 855), (500, 854)]}, {'description': 'CLIENTE', 'polygon': [(526, 843), (581, 847), (580, 861), (525, 857)]}, {'description': 'O', 'polygon': [(339, 850), (346, 850), (345, 864), (338, 864)]}, {'description': 'EN', 'polygon': [(356, 850), (374, 851), (373, 866), (355, 865)]}, {'description': 'WWW.SIRENA.DO', 'polygon': [(384, 852), (497, 859), (496, 874), (383, 867)]}, {'description': '.', 'polygon': [(501, 860), (505, 860), (504, 874), (500, 874)]}, {'description': 'GRACIAS', 'polygon': [(263, 865), (327, 869), (326, 885), (262, 881)]}, {'description': 'POR', 'polygon': [(337, 870), (363, 872), (362, 887), (336, 885)]}, {'description': 'SU', 'polygon': [(373, 872), (389, 873), (388, 888), (372, 887)]}, {'description': 'COMPRA', 'polygon': [(401, 873), (451, 876), (450, 892), (400, 889)]}, {'description': ',', 'polygon': [(455, 877), (457, 877), (456, 892), (454, 892)]}, {'description': 'VUELVA', 'polygon': [(472, 878), (523, 881), (522, 897), (471, 894)]}, {'description': 'PRONTO', 'polygon': [(535, 882), (582, 885), (581, 901), (534, 898)]}, {'description': 'VISA', 'polygon': [(407, 918), (441, 920), (440, 933), (406, 931)]}, {'description': 'TID', 'polygon': [(248, 930), (276, 931), (275, 947), (247, 946)]}, {'description': ':', 'polygon': [(278, 932), (284, 932), (283, 947), (277, 947)]}, {'description': '00030246', 'polygon': [(297, 932), (368, 935), (367, 951), (296, 948)]}, {'description': 'MID', 'polygon': [(416, 938), (440, 939), (439, 955), (415, 954)]}, {'description': ':', 'polygon': [(445, 940), (448, 940), (447, 955), (444, 955)]}, {'description': '000039200350106', 'polygon': [(462, 940), (589, 946), (588, 962), (461, 956)]}, {'description': 'VISA', 'polygon': [(246, 952), (282, 954), (281, 968), (245, 966)]}, {'description': 'DEBITO', 'polygon': [(295, 954), (348, 956), (347, 970), (294, 968)]}, {'description': 'COMPRA', 'polygon': [(541, 966), (591, 968), (591, 982), (541, 980)]}, {'description': 'HORA', 'polygon': [(469, 984), (501, 985), (500, 998), (468, 997)]}, {'description': '09:43:56', 'polygon': [(522, 986), (592, 989), (591, 1002), (521, 999)]}, {'description': '6', 'polygon': [(594, 659), (601, 660), (600, 669), (593, 668)]}, {'description': 'Codigo', 'polygon': [(225, 1329), (283, 1328), (283, 1345), (225, 1346)]}, {'description': 'de', 'polygon': [(296, 1328), (314, 1328), (314, 1345), (296, 1345)]}, {'description': 'Seguridad', 'polygon': [(327, 1328), (415, 1327), (415, 1344), (327, 1345)]}, {'description': ':', 'polygon': [(419, 1327), (422, 1327), (422, 1343), (419, 1343)]}, {'description': 'gPDP67', 'polygon': [(436, 1326), (492, 1325), (492, 1342), (436, 1343)]}, {'description': 'Fecha', 'polygon': [(223, 1354), (271, 1353), (271, 1370), (223, 1371)]}, {'description': 'de', 'polygon': [(284, 1353), (302, 1353), (302, 1369), (284, 1369)]}, {'description': 'Firma', 'polygon': [(315, 1352), (363, 1351), (363, 1368), (315, 1369)]}, {'description': 'Digital', 'polygon': [(376, 1350), (440, 1348), (440, 1365), (376, 1367)]}, {'description': ':', 'polygon': [(446, 1349), (449, 1349), (449, 1365), (446, 1365)]}, {'description': '21-09-2023', 'polygon': [(463, 1348), (557, 1346), (557, 1363), (463, 1365)]}, {'description': '09', 'polygon': [(565, 1346), (584, 1346), (584, 1362), (565, 1362)]}, {'description': ':', 'polygon': [(586, 1345), (590, 1345), (590, 1361), (586, 1361)]}, {'description': '46:59', 'polygon': [(222, 1380), (271, 1379), (271, 1392), (222, 1393)]}]}
CLUSTER_DATA = [{'description': 'SIRENA GALERIA 360', 'polygon': [(315, 115), (467, 115), (467, 140), (315, 140)]}, {'description': 'RNC 101796822 GAL.360 809-563-4111 GRUPO RAMOS S.A. RNC : 101796822', 'polygon': [(238, 132), (522, 132), (522, 184), (238, 184)]}, {'description': '21/09/23 09:46:58', 'polygon': [(239, 182), (385, 182), (385, 205), (239, 205)]}, {'description': 'e - NCF : E320004795244', 'polygon': [(241, 200), (403, 200), (403, 223), (241, 223)]}, {'description': 'FACTURA DE CONSUMO ELECTRONICA', 'polygon': [(277, 237), (526, 237), (526, 269), (277, 269)]}, {'description': 'DESCRIPCION', 'polygon': [(242, 271), (338, 271), (338, 289), (242, 289)]}, {'description': 'ABARROTES', 'polygon': [(245, 312), (321, 312), (321, 323), (245, 323)]}, {'description': '770241800484', 'polygon': [(246, 328), (350, 328), (350, 340), (246, 340)]}, {'description': 'MASTICAB FRE 841037603778 0.00 205.00 E', 'polygon': [(245, 346), (568, 346), (568, 379), (245, 379)]}, {'description': 'GALLETA CHOC PROD . FRESCOS 19.68 129.00', 'polygon': [(247, 382), (553, 382), (553, 418), (247, 418)]}, {'description': '000000011091', 'polygon': [(249, 423), (353, 423), (353, 436), (249, 436)]}, {'description': 'WRAP JAM QUE 074675028000 21.36 140.00', 'polygon': [(249, 442), (556, 442), (556, 472), (249, 472)]}, {'description': 'LUMIJOR PAN I 0.00 145.00 E', 'polygon': [(252, 478), (575, 478), (575, 504), (252, 504)]}, {'description': 'TOTAL A PAGAR VISA 41.04 619.00', 'polygon': [(253, 518), (560, 518), (560, 550), (253, 550)]}, {'description': 'ITBIS VALOR', 'polygon': [(408, 282), (547, 282), (547, 302), (408, 302)]}, {'description': 'Aprobacion : 310013', 'polygon': [(265, 1047), (437, 1047), (437, 1075), (265, 1075)]}, {'description': 'Monto RD $ : 619.00', 'polygon': [(265, 1070), (429, 1070), (429, 1098), (265, 1098)]}, {'description': 'ITBIS RDS : 41.04', 'polygon': [(266, 1093), (421, 1093), (421, 1121), (266, 1121)]}, {'description': 'FECHA : 21/09/2023', 'polygon': [(263, 985), (421, 985), (421, 1007), (263, 1007)]}, {'description': 'Modo de entrada : CHIP No. tarjeta : 459413 ****** 2743', 'polygon': [(263, 1003), (537, 1003), (537, 1053), (263, 1053)]}, {'description': 'ASHLEY GISSEL SOTO PUJOLS 250004449488', 'polygon': [(256, 577), (583, 577), (583, 601), (256, 601)]}, {'description': 'NUMERO ARTICULOS VENDIDOS = 4', 'polygon': [(294, 598), (539, 598), (539, 619), (294, 619)]}, {'description': '21/09/23 09:46 6019 30 0031 7188', 'polygon': [(286, 618), (558, 618), (558, 638), (286, 638)]}, {'description': 'ESTA COMPRA LE SUMA A SU BALANCE 6', 'polygon': [(258, 639), (585, 639), (585, 656), (258, 656)]}, {'description': 'PUNTOS .', 'polygon': [(393, 662), (449, 662), (449, 675), (393, 675)]}, {'description': 'Le Atendio : NICAURY A. SANTANA J', 'polygon': [(261, 679), (546, 679), (546, 694), (261, 694)]}, {'description': '619.00', 'polygon': [(513, 547), (561, 547), (561, 561), (513, 561)]}, {'description': '23092160190300071880031', 'polygon': [(345, 754), (516, 754), (516, 772), (345, 772)]}, {'description': 'VALORAMOS SU PREFERENCIA', 'polygon': [(324, 774), (530, 774), (530, 794), (324, 794)]}, {'description': 'ASHLEY GISSEL SOTO PUJOLS', 'polygon': [(316, 794), (533, 794), (533, 814), (316, 814)]}, {'description': 'CONSULTE NUESTRA POLITICA DE CAMBIOS', 'polygon': [(269, 811), (583, 811), (583, 835), (269, 835)]}, {'description': 'Y DEVOLUCIONES EN SERVICIO AL CLIENTE', 'polygon': [(259, 832), (585, 832), (585, 857), (259, 857)]}, {'description': 'O EN WWW.SIRENA.DO .', 'polygon': [(344, 853), (511, 853), (511, 875), (344, 875)]}, {'description': 'GRACIAS POR SU COMPRA , VUELVA PRONTO', 'polygon': [(270, 870), (591, 870), (591, 900), (270, 900)]}, {'description': 'VISA', 'polygon': [(419, 920), (454, 920), (454, 934), (419, 934)]}, {'description': 'TID : 00030246 MID : 000039200350106 COMPRA', 'polygon': [(262, 930), (609, 930), (609, 969), (262, 969)]}, {'description': 'VISA DEBITO HORA 09:43:56', 'polygon': [(263, 965), (611, 965), (611, 993), (263, 993)]}, {'description': 'Codigo de Seguridad : gPDP67 Digital : 21-09-2023 09 :', 'polygon': [(282, 1317), (648, 1317), (648, 1371), (282, 1371)]}, {'description': 'Fecha de Firma', 'polygon': [(283, 1357), (424, 1357), (424, 1391), (283, 1391)]}, {'description': '46:59', 'polygon': [(285, 1394), (335, 1394), (335, 1413), (285, 1413)]}]


def calculate_rotation_angle(data):
    angle = []
    rotation_angle = 0
    for item in data['values']:
        polygon = item['polygon']

        if len(polygon) < 2:
            return None

        # Extract the coordinates of the two upper points x,y
        point1 = (polygon[0][0], polygon[0][1])
        point2 = (polygon[1][0], polygon[1][1])

        # Calculate the angle of rotation (in degrees)
        delta_x = point2[0] - point1[0]
        delta_y = point2[1] - point1[1]
        rotation_angle = math.degrees(math.atan2(delta_y, delta_x))
        angle.append(rotation_angle)

    return np.median(angle)


def rotate_image(image_path, data,  angle):
    image = cv2.imread(image_path)
    height, width = image.shape[:2]
    center = (width // 2, height // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated_image = cv2.warpAffine(image, rotation_matrix, (width, height))
    # Iterate through each bounding box polygon and rotate it
    rotated_data = []
    for item in data['values']:
        rotated_polygon = []
        for point in item['polygon']:
            # Rotate each point using the same rotation matrix
            rotated_point = np.dot(rotation_matrix, [point[0], point[1], 1])
            rotated_polygon.append((int(rotated_point[0]), int(rotated_point[1])))
        # Add the rotated polygon to the list
        rotated_data.append({'description': item['description'], 'polygon': rotated_polygon})

    # draw_polygons(rotated_image, rotated_data)
    cluster_data = create_cluster(rotated_data)
    draw_polygons(rotated_image, cluster_data)
    cluster_data = create_cluster(cluster_data)
    draw_polygons(rotated_image, cluster_data)
    # create_cluster(cluster_data)



def draw_polygons(rotated_image, data):
    for item in data:
        polygon = item['polygon']
        polygon = np.array(polygon, np.int32)
        polygon = polygon.reshape((-1, 1, 2))

        # Draw the rotated polygon on the rotated image
        cv2.polylines(rotated_image, [polygon], isClosed=True, color=(0, 255, 0), thickness=2)

    cv2.imshow('Rotated Image with Polygons', rotated_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def join_words_with_spacing(clustered_polygons, clustered_words, median_height):
    result = []

    for i in range(len(clustered_words)):
        # Append the current word
        result.append(clustered_words[i])

        if i < len(clustered_words) - 1:
            # Calculate the distance between the current polygon and the next one
            current_polygon = clustered_polygons[i]
            next_polygon = clustered_polygons[i + 1]
            distance = abs(current_polygon[-1][0] - next_polygon[0][0])

            # Determine the spacing character based on the distance
            spacing_char = ' ' if distance <= median_height*4 else ' '*4
            result.append(spacing_char)

    return ''.join(result)


def create_cluster(rotated_data):
    print(rotated_data)
    # Calculate the median word height
    word_heights = [max(y for x, y in box['polygon']) - min(y for x, y in box['polygon']) for box in rotated_data]
    median_height = np.median(word_heights)

    # Calculate the median word y-centroid
    word_y_centroids = [(min(y for x, y in box['polygon']) + max(y for x, y in box['polygon'])) / 2 for box in
                        rotated_data]

    # Create a list of y-centroids for clustering
    y_centroids = np.array(word_y_centroids).reshape(-1, 1)


    # Set the epsilon parameter to the median word height
    epsilon = median_height/2

    # Initialize and fit the DBSCAN clustering algorithm
    dbscan = DBSCAN(eps=epsilon, min_samples=1)
    dbscan.fit_predict(y_centroids)

    # Get cluster labels (-1 represents outliers)
    labels = dbscan.labels_

    # Create a list to store the dictionaries with clustered words and rectangles
    clustered_results = []

    # Iterate through the word bounding boxes and cluster labels
    for cluster_label in np.unique(labels):

        # Get the indices of boxes in the current cluster
        cluster_indices = np.where(labels == cluster_label)[0]

        # Extract the words and word bounding boxes for the current cluster
        cluster_words = [rotated_data[i]['description'] for i in cluster_indices]
        cluster_polygons = [rotated_data[i]['polygon'] for i in cluster_indices]

        # Calculate the coordinates of the rectangle surrounding the cluster
        min_x = min(min(x for x, y in polygon) for polygon in cluster_polygons)
        min_y = min(min(y for x, y in polygon) for polygon in cluster_polygons)
        max_x = max(max(x for x, y in polygon) for polygon in cluster_polygons)
        max_y = max(max(y for x, y in polygon) for polygon in cluster_polygons)

        # Create a dictionary for the current cluster
        description = join_words_with_spacing(cluster_polygons, cluster_words, median_height)
        cluster_dict = {
            'description': description,  # Join words in the cluster
            'polygon': [(min_x, min_y), (max_x, min_y), (max_x, max_y), (min_x, max_y)]  # Rectangle coordinates
        }

        # Append the cluster dictionary to the results list
        clustered_results.append(cluster_dict)
    print(clustered_results)
    return clustered_results


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    image_path = 'resources/test2'
    response = ocr.OcrService().get_text(image_path)
    angle = calculate_rotation_angle(response)
    rotate_image(image_path, response, angle)
