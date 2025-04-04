import os
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split

class MNIST_Dataset(Dataset):
  def __init__(self, file_path, img_size, train=True):
    super(MNIST_Dataset, self).__init__()

    self.img_size = img_size
    self.file_path = file_path

    path=os.path.join(file_path, 'train.txt')
    if not train:
      path=os.path.join(file_path, 'test.txt')


    self.imgs=pd.read_csv(path, sep='\t', names=('image', 'label'), index_col=False)

    self.size=len(self.imgs)

  def __len__(self):
    return self.size

  def preprocess(self, image):
    img=np.asarray(image, dtype=np.float32) / 255.0
    return img

  def load(self, filename):
    return Image.open(filename)

  def __getitem__(self, idx):
    image_path=self.imgs['image'][idx]
    image_path=os.path.join(self.file_path, image_path)

    label=self.imgs['label'][idx]

    image=self.load(image_path)
    image=self.preprocess(image)

    return image, label