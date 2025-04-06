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

    self.sigmoid = nn.Sigmoid()

    self.softmax = nn.LogSoftmax(dim=1)

  def forward(self, x):
    x = self.flatten(x)
    x = self.input_layer(x)
    x = self.relu(self.hidden_layer(x))

    x = self.sigmoid(self.output_layer(x))
    #print(x)
    return x

#dataset1=MNIST_Dataset('skin_nskin.npy', (28, 28))

dataset=np.load('skin_nskin.npy')

epochs = 30
batch_size = 128
train_split = 0.8

n_train = int(len(dataset) * train_split)
n_val = len(dataset) - n_train
train_set, val_set = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(0))

loader_args = dict(batch_size=batch_size, num_workers=0, pin_memory=True)
train_loader = DataLoader(train_set, shuffle=True, **loader_args)
val_loader = DataLoader(val_set, shuffle=False, drop_last=True, **loader_args)

model = Simple_model(3, 2)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters())

#print(train_loader, n_train,n_val)

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
      #print(data[0])
      inputs=data
      labels = data #inputs = data[0:2], labels = data[-1]
      linput=[]
      llabel=[]
      #if i//10==0:
      print("We still going on",i)
      for j in range(len(inputs)):
        linput.append([int(inputs[j][0]),int(inputs[j][1]),int(inputs[j][2])])
        llabel.append([int(labels[j][-1])])
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
      labels = data #inputs = data[0:2], labels = data[-1]
      #print("i =",i)
      linput=[]
      llabel=[]
      for j in range(batch_size):
        linput.append([int(inputs[j][0]),int(inputs[j][1]),int(inputs[j][2])])
        llabel.append([int(labels[j][-1])])
      inputs=torch.tensor(linput)
      inputs = inputs.to(device=device, dtype=torch.float32)
      labels = torch.tensor(llabel)
      labels = labels.to(device=device, dtype=torch.float32)

      outputs = model(inputs)
      val_loss += loss_fn(outputs, labels).item()/len(val_loader)
      #print("we reached acc")
      for i in range(len(outputs)):
        if round(float(outputs[i]),1) == labels[i]:
          acc += 1
      #print("we got past acc")

    epoch_accuracy.append(acc/len(val_loader))
    val_losses.append(val_loss)
print(losses)
print(val_losses)
print(epoch_accuracy)

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