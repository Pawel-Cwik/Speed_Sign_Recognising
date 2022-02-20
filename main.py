import numpy as np
import argparse
import cv2

import xml.etree.ElementTree as ET
import os
import shutil

if os.path.exists("train"):
    print("Istnieje")
else:
    os.mkdir("train")
if os.path.exists("test"):
    print("Istnieje")
else:
    os.mkdir("test")


path = "../annotations"
for filename in os.listdir(path):
    if not filename.endswith('.xml'): continue
    fullname = os.path.join(path, filename)
    tree = ET.parse(fullname)
    root = tree.getroot()
    for child in root.findall('object'):
         name= child.find('name').text
         if name == 'speedlimit':
             shutil.copy(f'../annotations/{filename}', "./test")
#             print(True)









