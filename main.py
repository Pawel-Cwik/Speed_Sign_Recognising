import numpy as np
import cv2

import xml.etree.ElementTree as ET
import os
import shutil
from pathlib import PosixPath, Path
import sklearn
from sklearn.ensemble import RandomForestClassifier

# TODO Jakość kodu i raport (4/4)


# TODO Skuteczność klasyfikacji 0.925 (4/4)
# TODO [0.00, 0.50) - 0.0
# TODO [0.50, 0.55) - 0.5
# TODO [0.55, 0.60) - 1.0
# TODO [0.60, 0.65) - 1.5
# TODO [0.65, 0.70) - 2.0
# TODO [0.70, 0.75) - 2.5
# TODO [0.75, 0.80) - 3.0
# TODO [0.80, 0.85) - 3.5
# TODO [0.85, 1.00) - 4.0

# TODO Skuteczność detekcji (/2)

# TODO Poprawki po terminie. (-1)

def check_folders():
# TODO Te katalogi juz powinny istniec.
    if os.path.exists("../train"):

        if not os.path.exists("../train/annotations"):
            os.mkdir("../train/annotations")
            os.mkdir("../train/images")
    else:
        os.mkdir("../train")
        os.mkdir("../train/annotations")
        os.mkdir("../train/images")

    if os.path.exists("../test"):

        if not os.path.exists("../test/annotations"):
            os.mkdir("../test/annotations")
            os.mkdir("../test/images")
    else:
        os.mkdir("../test")
        os.mkdir("../test/annotations")
        os.mkdir("../test/images")


# 0 - speedlimit
# 1 - other
class Recognising:
    def __init__(self, path_to_file=None):
        self._path_to_file = path_to_file
        if self._path_to_file != None:
            # TODO Nadal zla sciezka! Poprawione przeze mnie. (-1)
            self._image = Path(f'../train/images/{path_to_file.name.rstrip(".xml")}.png')
            self._type = []
            self._bnd = []
            self._shape = []
            self._number_of_objects = None
            self._descriptor = []
            self._predict_arg = []

            self.read_file()

    def create(self, name, no, bnd):
        self._image = Path(f'../test/images/{name}')
        self._path_to_file = Path(f'../test/annotations/{name.rstrip(".png")}.xml')
        self._bnd = [list(map(int, i)) for i in bnd]
        self._number_of_objects = no
        self._predict_arg = []
        self._descriptor = []
        self._shape = []
        if self._path_to_file.is_file():
            tree = ET.parse(self._path_to_file)
            root = tree.getroot()
            names = []
            size = root.find('size')
            self._shape = [size.find('width').text, size.find('height').text]
            for i in range(len(self._bnd)):
                if int(self._shape[0]) * 1 / 10 > self._bnd[i][1] - self._bnd[i][0] or int(self._shape[1]) * 1 / 10 > \
                        self._bnd[i][3] - self._bnd[i][2]:
                    self._bnd[i] = [0, 1, 0, 1]

    def read_file(self):
        if self._path_to_file.is_file():
            tree = ET.parse(self._path_to_file)
            root = tree.getroot()
            names = []
            size = root.find('size')
            self._shape = [size.find('width').text, size.find('height').text]
            for child in root.findall('object'):
                name = child.find('name').text
                names.append(name)
                if name == 'speedlimit':
                    self._type.append(0)
                else:
                    self._type.append(1)

                bbox = child.find('bndbox')
                self._bnd.append([int(bbox.find('xmin').text), int(bbox.find('xmax').text),
                                  int(bbox.find('ymin').text), int(bbox.find('ymax').text)])

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

        if 'speedlimit' in names:

            if iter_speedlimit < 3:
                shutil.copy(f'{filepath}', "../train/annotations")
                shutil.copy(f'../images/{filepath.name.rstrip(".xml")}.png', "../train/images")
                iter_speedlimit = iter_speedlimit + 1

            else:
                shutil.copy(f'{filepath}', "../test/annotations")
                shutil.copy(f'../images/{filepath.name.rstrip(".xml")}.png', "../test/images")
                iter_speedlimit = 0

        else:
            if iter_other < 3:
                shutil.copy(f'{filepath}', "../train/annotations")
                shutil.copy(f'../images/{filepath.name.rstrip(".xml")}.png', "../train/images")
                iter_other = iter_other + 1
            else:
                shutil.copy(f'{filepath}', "../test/annotations")
                shutil.copy(f'../images/{filepath.name.rstrip(".xml")}.png', "../test/images")
                iter_other = 0


def learning(elements: list):
    dictionarySize = 100
    sift = cv2.SIFT_create()

    BOW = cv2.BOWKMeansTrainer(dictionarySize)

    for ele in elements:
        full_image = cv2.imread(str(ele._image))
        for i in range(ele._number_of_objects):
            image = full_image[ele._bnd[i][2]:ele._bnd[i][3], ele._bnd[i][0]:ele._bnd[i][1], :]
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            kp, dsc = sift.detectAndCompute(gray, None)
            if dsc is not None:
                BOW.add(dsc)

    # dictionary created
    dictionary = BOW.cluster()
    np.save("dictionary.npy",dictionary)   #odkomentować tylko za pierwszym uruchomieniem programu


def extracting(elements: list):
    sift = cv2.SIFT_create()
    flann = cv2.FlannBasedMatcher_create()
    bow = cv2.BOWImgDescriptorExtractor(sift, flann)
    dictionary = np.load('dictionary.npy')
    bow.setVocabulary(dictionary)

    for ele in elements:
        full_image = cv2.imread(str(ele._image))
        for i in range(ele._number_of_objects):
            image = full_image[ele._bnd[i][2]:ele._bnd[i][3], ele._bnd[i][0]:ele._bnd[i][1], :]
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            descriptor = bow.compute(gray, sift.detect(gray))
            if descriptor is not None:
                ele._descriptor.append(descriptor)
            else:
                # TODO Lepiej w ogole pominac takie przypadki.
                ele._descriptor.append(np.zeros((1, 100)))


def learn(elements: list):
    descriptors = np.empty((1, 100))
    labels = []
    classify = RandomForestClassifier(n_estimators=100)
    for elem in elements:
        # if elem._number_of_objects > 0:
        for i in range(elem._number_of_objects):
            labels.append(elem._type[i])
            descriptors = np.vstack((descriptors, elem._descriptor[i]))
    classify.fit(descriptors[1:], labels)
    return classify


def prediction_img(elements: list, op):
    for elems in elements:
        for i in range(elems._number_of_objects):
            elems._predict_arg.append(op.predict(elems._descriptor[i]))


def terminal_serve():
    list_of_input = []
    x = input()
    if x == "classify":
        it = 0

        it = int(input())
        for i in range(it):
            name = None
            parts = 0
            bnd = []
            name = str(input())
            parts = int(input())
            for j in range(parts):
                bnd.append(input().split(" "))
            obj = Recognising()
            obj.create(name=name, no=parts, bnd=bnd)
            list_of_input.append(obj)
    return list_of_input


def output(out):
    for x in out:
        if len(x._predict_arg) != 0:
            for o in x._predict_arg:
                if o == 1:
                    print("other")
                elif o == 0:
                    print("speedlimit")
        else:
            print("other")


if __name__ == '__main__':
    # ann_path = Path("../annotations")
    # check_folders()

    list_of_elements = []
    # if ann_path.is_dir():
    #     for file in list(ann_path.glob('*.xml')):
    #         make_folders(file)
# TODO Zla sciezka.
dict_path = Path("../train/annotations")
if dict_path.is_dir():
    for file in list(dict_path.glob('*.xml')):
        list_of_elements.append(Recognising(Path(file)))

learning(list_of_elements)
extracting(list_of_elements)
ops = learn(list_of_elements)

# list_of_test_elements = []
# test_path = Path("./test/annotations")
# if test_path.is_dir():
#     for files in list(test_path.glob('*.xml')):
#         list_of_test_elements.append(Recognising(Path(files)))
#
# extracting(list_of_test_elements)
# prediction_img(list_of_test_elements, ops)

list_of_inputs = terminal_serve()
extracting(list_of_inputs)
prediction_img(list_of_inputs, ops)
output(list_of_inputs)

# for el in list_of_test_elements:
#     image = cv2.imread(str(el._image))
#     cv2.imshow(str(el._predict_arg), image)
#     cv2.waitKey(0)
