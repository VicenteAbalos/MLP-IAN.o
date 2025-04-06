import os
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split, WeightedRandomSampler

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

    #"""
    self.input_layer = nn.Linear(input_size, 8)

    self.hidden_layer = nn.Linear(8, 16)

    self.output_layer = nn.Linear(16, n_classes)
    #"""
    """
    self.input_layer = nn.Linear(input_size, 4)

    self.hidden_layer = nn.Linear(4, 8)

    self.output_layer = nn.Linear(8, n_classes)
    """

    self.relu = nn.ReLU()

    self.sigmoid = nn.Sigmoid()

    self.softmax = nn.LogSoftmax(dim=1)

  def forward(self, x):
    x = self.flatten(x)
    x = self.input_layer(x)

    x = self.relu(self.hidden_layer(x))
    #x = self.sigmoid(self.hidden_layer(x))

    x = self.sigmoid(self.output_layer(x))
    #x = self.softmax(self.output_layer(x))
    return x

def mirror(n):
  if n==1:
    return 0
  elif n==0:
    return 1
  
def get_round(out):
  return (round(float(out[0]),1),round(float(out[1]),1))

dataset=np.load('skin_nskin.npy')

epochs = 30
batch_size = 128
train_split = 0.8

n_train = int(len(dataset) * train_split)
n_val = len(dataset) - n_train
train_set, val_set = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(0))

class_weights=[1,1]
train_weight=[0]*len(train_set)
val_weight=[0]*len(val_set)

for idx in range(len(train_set)):
  c_train_weight=class_weights[train_set[idx][3]]
  train_weight[idx]=c_train_weight
for idx in range(len(val_set)):
  c_val_weight=class_weights[val_set[idx][3]]
  val_weight[idx]=c_val_weight

t_sampler=WeightedRandomSampler(train_weight,num_samples=len(train_weight),replacement=True)
v_sampler=WeightedRandomSampler(val_weight,num_samples=len(val_weight),replacement=True)

loader_args = dict(batch_size=batch_size, num_workers=0, pin_memory=True)
train_loader = DataLoader(train_set, sampler=t_sampler, **loader_args)
val_loader = DataLoader(val_set, sampler=v_sampler, drop_last=True, **loader_args)
model = Simple_model(3, 1)
#model = Simple_model(3, 2)

loss_fn = nn.BCELoss()
#loss_fn = nn.CrossEntropyLoss()

optimizer = torch.optim.SGD(model.parameters())


device = 'cpu'
losses = []
val_losses = []
epoch_accuracy = []

model.to(device)

model.train()

for epoch in range(epochs):
  acc = 0.0
  val_loss = 0.0
  epoch_loss = 0.0

  for i, data in enumerate(train_loader):
      inputs=data
      labels = data
      linput=[]
      llabel=[]
      for j in range(len(inputs)):
        linput.append([int(inputs[j][0]),int(inputs[j][1]),int(inputs[j][2])])

        llabel.append([int(labels[j][-1])])
        #llabel.append([int(labels[j][-1]),mirror(int(labels[j][-1]))])

      inputs=torch.tensor(linput)
      inputs = inputs.to(device=device, dtype=torch.float32)
      labels = torch.tensor(llabel)
      labels = labels.to(device=device, dtype=torch.float32)

      optimizer.zero_grad()
      outputs = model(inputs)

      loss = loss_fn(outputs, labels)
      loss.backward()
      optimizer.step()

      epoch_loss += loss.item()/len(train_loader)

  losses.append(epoch_loss)

  with torch.no_grad():
    for i, data in enumerate(val_loader):
      inputs=data
      labels = data
      linput=[]
      llabel=[]
      for j in range(batch_size):
        linput.append([int(inputs[j][0]),int(inputs[j][1]),int(inputs[j][2])])

        llabel.append([int(labels[j][-1])])
        #llabel.append([int(labels[j][-1]),mirror(int(labels[j][-1]))])

      inputs=torch.tensor(linput)
      inputs = inputs.to(device=device, dtype=torch.float32)
      labels = torch.tensor(llabel)
      labels = labels.to(device=device, dtype=torch.float32)

      outputs = model(inputs)
      val_loss += loss_fn(outputs, labels).item()/len(val_loader)
      for i in range(len(outputs)):
        if round(float(outputs[i]),1) == labels[i]:
        #if get_round(outputs[i])[0] == labels[i][0] and get_round(outputs[i])[1] == labels[i][1]:
          acc += 1

    epoch_accuracy.append(acc/len(val_loader))
    val_losses.append(val_loss)

"""print(losses)
print(val_losses)
print(epoch_accuracy)"""

plt.plot(losses, label='Train Loss')
plt.plot(val_losses, label='Validation Loss')
plt.legend()
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()

plt.plot(epoch_accuracy, label='Accuracy')
plt.legend()
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.show()