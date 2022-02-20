import numpy as np
import argparse
import cv2

import xml.etree.ElementTree as ET
import os
import shutil
from pathlib import PosixPath, Path


def check_folders():
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


# 0 - speedlimit
# 1 - other
class Recognising:
    def __init__(self, path_to_file: Path):
        self._path_to_file = path_to_file
        self._image = Path(f'../images/{path_to_file.name.rstrip(".xml")}.png')
        self._type = []
        self._bnd = []
        self._shape = []
        self._number_of_objects = None
        self.descriptor = []

        self.read_file()  # to musi być ostatnie @@@@@@@@@@@@@@@@@@@@@@@@@@@@

    def read_file(self):
        if self._path_to_file.is_file():
            tree = ET.parse(self._path_to_file)
            root = tree.getroot()
            names = []
            size = root.find('size')
            self._shape = [size.find('width'), size.find('height')]
            for child in root.findall('object'):
                name = child.find('name').text
                names.append(name)
                if name == 'speedlimit':
                    self._type.append(0)
                else:
                    self._type.append(1)

                bbox = child.find('bndbox')
                self._bnd.append([int(bbox.find('xmin').text), int(bbox.find('ymin').text),
                                  int(bbox.find('xmax').text), int(bbox.find('ymax').text)])

            self._number_of_objects = len(names)


iter_speedlimit = 0
iter_other = 0


def make_folders(filepath):
    global iter_other, iter_speedlimit
    if filepath.is_file():

        tree = ET.parse(filepath)
        root = tree.getroot()
        names = []
        for child in root.findall('object'):
            names.append(child.find('name').text)
        # print(names)

        if 'speedlimit' in names:
            # print("IM HERE")
            if iter_speedlimit < 3:
                shutil.copy(f'{filepath}', "./train/annotations")
                shutil.copy(f'../images/{filepath.name.rstrip(".xml")}.png', "./train/images")
                iter_speedlimit = iter_speedlimit + 1

            else:
                shutil.copy(f'{filepath}', "./test/annotations")
                shutil.copy(f'../images/{filepath.name.rstrip(".xml")}.png', "./test/images")
                iter_speedlimit = 0

        else:
            if iter_other < 3:
                shutil.copy(f'{filepath}', "./train/annotations")
                shutil.copy(f'../images/{filepath.name.rstrip(".xml")}.png', "./train/images")
                iter_other = iter_other + 1
            else:
                shutil.copy(f'{filepath}', "./test/annotations")
                shutil.copy(f'../images/{filepath.name.rstrip(".xml")}.png', "./test/images")
                iter_other = 0



def learning(elements: list):
    dictionarySize = 100
    sift = cv2.SIFT_create()

    BOW = cv2.BOWKMeansTrainer(dictionarySize)

    for ele in elements:
        full_image = cv2.imread(str(ele._image))
        for i in range(ele._number_of_objects):
            image = full_image[ele._bnd[i][1]:ele._bnd[i][3], ele._bnd[i][0]:ele._bnd[i][2], :]
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            kp, dsc = sift.detectAndCompute(gray, None)
            if dsc is not None:
                BOW.add(dsc)

    # dictionary created
    dictionary = BOW.cluster()
    # np.save("dictionary.npy",dictionary)   #odkomentować tylko za pierwszym uruchomieniem programu


def extracting(elements: list):
    sift = cv2.SIFT_create()
    flann = cv2.FlannBasedMatcher_create()
    bow = cv2.BOWImgDescriptorExtractor(sift, flann)
    dictionary = np.load('dictionary.npy')
    bow.setVocabulary(dictionary)

    for ele in elements:
        full_image = cv2.imread(str(ele._image))
        for i in range(ele._number_of_objects):
            image = full_image[ele._bnd[i][1]:ele._bnd[i][3], ele._bnd[i][0]:ele._bnd[i][2], :]
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            descriptor = bow.compute(gray, sift.detect(gray))
            ele.descriptor.append(descriptor)


if __name__ == '__main__':
    ann_path = Path("../annotations")
    check_folders()
    list_of_elements = []
    if ann_path.is_dir():
        for file in list(ann_path.glob('*.xml')):
            make_folders(file)
    dict_path = Path("./train/annotations")
    if dict_path.is_dir():
        for file in list(dict_path.glob('*.xml')):
            list_of_elements.append(Recognising(Path(file)))
# learning(list_of_elements)
# extracting(list_of_elements)


