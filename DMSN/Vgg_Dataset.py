import torch
import torch.utils.data as data
import csv
import pandas as pd

from PIL import Image


class ClipRecord(object):
    def __init__(self, row):
        self._data = row

    @property
    def path(self):
        return self._data[0]

    # @property
    # def start_frames(self):
    #     return int(self._data[1])

    @property
    def label(self):
        return 1
    #
    #     with open("/home/cike/pythonGC/P3D/VGG_Face2/identity_meta.csv", "rt") as csvfile:
    #         reader = csv.reader("/home/cike/pythonGC/P3D/VGG_Face2/identity_meta.csv")
    #         class_id = []
    #         flag = []
    #         for row in reader:
    #             class_id.append(row[0])
    #             flag.append(row[3])
    #         class_id.pop(0)
    #         flag.pop(0)
    #     paths = self._data[0].split('/')
    #     for i in range(len(class_id)):
    #         if class_id[i] == paths[0]:
    #             return  int(flag[i])


class VggDataset(data.Dataset):
    def __init__(self, list_file, length=16, modality='RGB', image_tmpl='{:d}.jpg', transform=None, type='train'):

        self.list_file = list_file
        self.length = length
        self.modality = modality
        self.image_tmpl = image_tmpl
        self.transform = transform
        self.type = type

        self._parse_list()

    def _load_image(self, directory):
        if self.modality == 'RGB':
            return Image.open(directory)

    # def _load_image(self, directory):
    #     if self.modality == 'RGB':
    #         return Image.open(os.path.join("/home/cike/pythonGC/P3D/VGG_Face/train" + directory))

    def _parse_list(self):
        data1 = pd.read_csv('/home/cike/pythonGC/P3D/VGG_Face2/identity_meta.csv', encoding='ISO-8859-1',
                            usecols=['Class_ID', ' Gender'])
        self.class_list = data1['Class_ID']
        self.label_list = data1[' Gender']
        self.clip_list = [ClipRecord(x.strip().split(' ')) for x in open(self.list_file)]

    def __getitem__(self, index):
        record = self.clip_list[index]
        return self.get(record)

    def get(self, record):
        clip = list()
        if self.type == 'train':
            img_path = '/home/cike/pythonGC/P3D/VGG_Face2/train/' + record.path
        else:
            img_path = '/home/cike/pythonGC/P3D/VGG_Face2/test/' + record.path
        img = self._load_image(img_path)
        paths = record.path.split('/')
        # img = self._load_image(record.path)
        img = img.resize((160, 160))
        for index in range(len(self.class_list)):
            if self.class_list[index] == paths[0]:
                label = self.label_list[index]
                if label == ' m':
                    label = 0
                else:
                    label = 1
        for i in range(self.length):
            clip.append(img)
        clip = self.transform(clip)

        return clip, label

    def __len__(self):
        return len(self.clip_list)
