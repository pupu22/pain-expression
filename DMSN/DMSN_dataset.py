import torch
import torch.utils.data as data

from PIL import Image
import os
import os.path
import numpy as np


class ClipRecord(object):
    def __init__(self, row):
        self._data = row

    @property
    def path(self):
        return "/home/cike/pythonGC/UNBC" + self._data[0]

    @property
    def start_frames(self):
        return int(self._data[1])

    @property
    def subject(self):
        return self._data[0].split('/')[-3]

    @property
    def label(self):
        # paths = self._data[0].split('/')
        label = int(self._data[2])
        if label == 0:
            return 0
        elif label == 1:
            return 1
        elif label == 2:
            return 2
        elif label == 3:
            return 3
        elif label <= 5:
            return 4
        elif label > 5:
            return 5


class DMSNDataSet(data.Dataset):
    def __init__(self, list_file, length=16, modality='RGB', image_tmpl='{:d}.jpg', transform=None, data_type='train',
                 index=0):

        self.list_file = list_file
        self.length = length
        self.modality = modality
        self.image_tmpl = image_tmpl
        self.transform = transform
        self.data_type = data_type
        self.index = index

        self._parse_list()

    def _load_image(self, directory, name, idx):
        if self.modality == 'RGB':
            idx_3bite = "%03d" % idx
            filename = name + idx_3bite + ".png"
            return Image.open(os.path.join(directory, filename)).convert('RGB')
            # return Image.open(os.path.join(directory, self.image_tmpl.format(idx)))

    # def _load_image(self, directory):
    #     if self.modality == 'RGB':
    #         return Image.open(os.path.join("/home/cike/pythonGC/P3D/VGG_Face/train" + directory))

    def _parse_list(self):
        self.clip_list = [ClipRecord(x.strip().split(' ')) for x in open(self.list_file)]
        self.sub_list = list(set([ClipRecord(x.strip().split(' ')).subject for x in open(self.list_file)]))
        if self.data_type == 'train':
            self.sub_list.pop(self.index)
            print(self.sub_list)
            clip_temp = []
            for i in range(len(self.clip_list)):
                if self.clip_list[i].subject in self.sub_list:
                    clip_temp.append(self.clip_list[i])

            self.clip_list = clip_temp
        else:
            self.sub_list = [self.sub_list[self.index]]
            clip_temp = []
            for i in range(len(self.clip_list)):
                if self.clip_list[i].subject in self.sub_list:
                    clip_temp.append(self.clip_list[i])
            self.clip_list = clip_temp

    def __getitem__(self, index):
        record = self.clip_list[index]
        return self.get(record)

    def get(self, record):
        clip = list()
        for i in range(self.length):
            name = record.path.split('/')
            img = self._load_image(record.path, name[-2], i + record.start_frames)
            # img = self._load_image(record.path)\
            img = img.resize((160, 160))
            clip.append(img)

        clip = self.transform(clip)

        return clip, record.label

    def __len__(self):
        return len(self.clip_list)
