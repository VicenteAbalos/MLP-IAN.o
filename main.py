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
  
class Simple_model(nn.Module):
  def __init__(self, input_size, n_classes):
    super(Simple_model, self).__init__()

    self.flatten = nn.Flatten()

    self.input_layer = nn.Linear(input_size, 8)

    self.hidden_layer = nn.Linear(8, 16)

    self.output_layer = nn.Linear(16, n_classes)

    self.relu = nn.ReLU()

  def forward(self, x):
    x = self.flatten(x)
    x = self.relu(self.input_layer(x))

    x = self.relu(self.hidden_layer(x))

    x = self.output_layer(x)
    return x

#dataset=MNIST_Dataset('skin_nskin.npy', (28, 28))

dataset=np.load('skin_nskin.npy')

epochs = 10
batch_size = 1
train_split = 0.8

n_train = int(len(dataset) * train_split)
n_val = len(dataset) - n_train
train_set, val_set = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(0))

loader_args = dict(batch_size=batch_size, num_workers=0, pin_memory=True)
train_loader = DataLoader(train_set, shuffle=True, **loader_args)
val_loader = DataLoader(val_set, shuffle=False, drop_last=True, **loader_args)

model = Simple_model(3, 1)
loss_fn = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters())

print(dataset[0], n_train,n_val)