import numpy as np
import argparse
import cv2

import xml.etree.ElementTree as ET
import os
import shutil

if os.path.exists("train"):
    print("Istnieje")
    if not os.path.exists("train/annotations"):
        os.mkdir("train/annotations")
        os.mkdir("train/images")
else:
    os.mkdir("train")
    os.mkdir("train/annotations")
    os.mkdir("train/images")

if os.path.exists("test"):
    print("Istnieje")
    if not os.path.exists("test/annotations"):
        os.mkdir("test/annotations")
        os.mkdir("test/images")
else:
    os.mkdir("test")
    os.mkdir("test/annotations")
    os.mkdir("test/images")

iter_split= 0
iter_traffic=0
iter_stop=0
iter_cross=0


ann_path = "../annotations"
img_path= "../images"
_list=os.listdir(ann_path)
for filename in _list:
    if not filename.endswith('.xml'): continue
    fullname = os.path.join(ann_path, filename)
    tree = ET.parse(fullname)
    root = tree.getroot()
    for child in root.findall('object'):
         name= child.find('name').text
         name_split = os.path.splitext(filename)[0]  # wyciągnięcie nazwy pliku żeby użyć jej z plikami png
         if name == 'speedlimit':

             if iter_split < 3:
                 shutil.copy(f'../annotations/{filename}', "./train/annotations")
                 shutil.copy(f'../images/{name_split}.png', "./train/images")
                 iter_split= iter_split+1

             else:
                 shutil.copy(f'../annotations/{filename}', "./test/annotations")
                 shutil.copy(f'../images/{name_split}.png', "./test/images")
                 iter_split =0
         if name == 'trafficlight':

                 if iter_traffic < 3:
                     shutil.copy(f'../annotations/{filename}', "./train/annotations")
                     shutil.copy(f'../images/{name_split}.png', "./train/images")
                     iter_traffic = iter_traffic + 1

                 else:
                     shutil.copy(f'../annotations/{filename}', "./test/annotations")
                     shutil.copy(f'../images/{name_split}.png', "./test/images")
                     iter_traffic = 0
         if name == 'stop':

                 if iter_stop < 3:
                     shutil.copy(f'../annotations/{filename}', "./train/annotations")
                     shutil.copy(f'../images/{name_split}.png', "./train/images")
                     iter_stop = iter_stop + 1

                 else:
                     shutil.copy(f'../annotations/{filename}', "./test/annotations")
                     shutil.copy(f'../images/{name_split}.png', "./test/images")
                     iter_stop = 0

         if name == 'crosswalk':

                 if iter_cross < 3:
                     shutil.copy(f'../annotations/{filename}', "./train/annotations")
                     shutil.copy(f'../images/{name_split}.png', "./train/images")
                     iter_cross = iter_cross + 1

                 else:
                     shutil.copy(f'../annotations/{filename}', "./test/annotations")
                     shutil.copy(f'../images/{name_split}.png', "./test/images")
                     iter_cross = 0










