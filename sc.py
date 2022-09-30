#!/usr/bin/env python
# coding: utf-8

# In[31]:


import torch
import torch.nn as nn
import h5py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset,DataLoader
from torch.utils.tensorboard import SummaryWriter
from pytorch_transformers import AdamW, WarmupLinearSchedule, WarmupConstantSchedule
import torch.nn.functional as F
import random
import time
import os
import cv2
from PIL import Image
from tqdm import tqdm
img_path=r'D:\Python\Python_Project\NJU\Single Modality\data\MVSA\MVSA_Single\data'


# In[2]:


def crop_centre(img,new_width,new_height):
    height,width,_=img.shape
    startx=width//2-new_width//2
    starty=height//2-new_height//2
    return img[starty:starty+new_height,startx:startx+new_width, :]


# In[3]:


'''
Since the images vary in size, you first need to resize the images to the same size
'''
def image_processing(img_path):
    for dir_image in tqdm(os.listdir(img_path)):
        if dir_image[-3:]=='jpg':
            img_p=Image.open(os.path.join(img_path,dir_image))
            img_size=img_p.resize((512,512))
            img_size.save(fr'D:\Python\Python_Project\NJU\Single Modality\data\MVSA\MVSA_Single\img_resize\{dir_image}','JPEG')

# image_processing(img_path)


# In[4]:


img_processed=r'D:\Python\Python_Project\NJU\Single Modality\data\MVSA\MVSA_Single\img_resize'
def save_image_h5(img_path=img_path):
    img_list=[]
    for dir_image in tqdm(os.listdir(img_path)):
        if dir_image[-3:]=='jpg':
            img=cv2.imread(os.path.join(img_path,dir_image))
            img=crop_centre(img,512,512)
            img_list.append(img)

    img_np = np.array(img_list)
    with h5py.File(r'D:\Python\Python_Project\NJU\Single Modality\data\MVSA\MVSA_Single\image.h5','w') as f:
        f.create_dataset('img_data',data = img_np)
        f.close()
    print('Save Successfully...')
# save_image_h5(img_path=img_processed)


# In[14]:


def normalize(image):
    mean = np.mean(image)
    var = np.mean(np.square(image-mean))
    image = (image - mean)/np.sqrt(var)
    return image
class ImgDataset(Dataset):
    def __init__(self,h5_path,csv_name='Label.csv'):

        self.file_object=h5py.File(h5_path,'r')
        self.dataset=self.file_object['img_data']
        self.label_df=pd.read_csv(csv_name)
        pass
    def __len__(self):
        return len(self.dataset)
    def __getitem__(self, item):
        if(item>len(self.dataset)):
            raise IndexError()
        img_value=torch.FloatTensor(np.array(self.dataset[int(item)]))
        img_label=torch.LongTensor([self.label_df['image_label'][item]])
        return img_value,img_label
    def plot_image(self,index):
        arr=np.array((self.dataset[int(index)])/255.0)
        plt.imshow(arr,interpolation='nearest')
        plt.show()
#The role of View is to tile the (218,178,3) three-dimensional tensor into a one-dimensional tensor 218*178*3
batchsize=16
class View(nn.Module):
    def __init__(self, shape):
        super().__init__()
        self.shape = shape,

    def forward(self, x):
        return x.view(*self.shape)


# In[6]:


imgdataset=ImgDataset(h5_path=r'D:\Python\Python_Project\NJU\Single Modality\data\MVSA\MVSA_Single\image.h5')


# In[7]:


# imgdataset.plot_image(2)
# 


# In[8]:


# imgdataset.plot_image(8)


# In[10]:


data_loader=DataLoader(imgdataset,batch_size=batchsize,shuffle=True)

# In[32]:


class Classifier(nn.Module):
    def __init__(self):
        super().__init__()

        # Conv2d[ channels, output, height_2, width_2 ]
        self.layer1 = nn.Sequential(nn.Conv2d(3, 256, kernel_size=5, stride=2),
                                    # Feature map size 225x225
                                    nn.BatchNorm2d(256),
                                    nn.LeakyReLU(0.02))
        self.layer2 = nn.Sequential(nn.Conv2d(256, 64, kernel_size=4, stride=2),
                                    # Feature map size 81x81
                                    nn.BatchNorm2d(64),
                                    nn.LeakyReLU(0.02))
        self.layer3 = nn.Sequential(nn.Conv2d(64, 3, kernel_size=3, stride=2),
                                    # Feature map size 9x9
                                    nn.LeakyReLU(0.2))

        self.layer4=nn.Sequential(nn.Linear(3 * 30 * 30,1024),
                                  nn.Linear(1024,512),
                                  nn.Dropout(0.5),
                                  nn.Linear(512,64),
                                  nn.BatchNorm1d(64),
                                  nn.Linear(64,32),
                                  nn.Dropout(0.5),
                                  nn.Linear(32,16),
                                  nn.Dropout(0.5),
                                  nn.Linear(16,3))

    def forward(self,inputs):
        conv1=self.layer1(inputs)

        conv2=self.layer2(conv1)

        conv3=self.layer3(conv2)
        conv3=conv3.view(conv3.shape[0],-1)
        pred=self.layer4(conv3)



        return pred
class AlexNet(nn.Module):
    def __init__(self,num_classes=3):
        super(AlexNet,self).__init__()
        self.feature_extraction = nn.Sequential(
            nn.Conv2d(in_channels=3,out_channels=96,kernel_size=11,stride=4,padding=2,bias=False),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3,stride=2,padding=0),
            nn.Conv2d(in_channels=96,out_channels=192,kernel_size=5,stride=1,padding=2,bias=False),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3,stride=2,padding=0),
            nn.Conv2d(in_channels=192,out_channels=384,kernel_size=3,stride=1,padding=1,bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=384,out_channels=256,kernel_size=3,stride=1,padding=1,bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=256,out_channels=256,kernel_size=3,stride=1,padding=1,bias=False),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=0),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(in_features=256*7*7,out_features=4096),
            nn.Dropout(p=0.5),
            nn.Linear(in_features=4096, out_features=4096),
            nn.Linear(in_features=4096, out_features=num_classes),
        )
    def forward(self,x):
        x = self.feature_extraction(x)
        x = x.view(x.size(0),256*7*7)
        x = self.classifier(x)
        return x
def categorical_accuracy(preds, y):
    """
    Returns accuracy per batch, i.e. if you get 8/10 right, this returns 0.8, NOT 8
    """
    top_pred = preds.argmax(1, keepdim = True)
    correct = top_pred.eq(y.view_as(top_pred)).sum()
    acc = correct.float() / y.shape[0]
    return acc


# In[33]:


#parameter:
DEVICE='cuda'
LR=0.01
N_EPOCH=50
print('Build Model...')
model=AlexNet()
model.load_state_dict(torch.load('49CNN-fc-model.pt'))
model=model.to(DEVICE)
optimizer=torch.optim.SGD(model.parameters(),lr=LR)
scheduler=WarmupConstantSchedule(optimizer,warmup_steps=10*len(data_loader))
lossfunction=nn.CrossEntropyLoss()
print('Write log...')
writer=SummaryWriter()
def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


# In[34]:



for i in range(N_EPOCH):
    each_loss=0
    acc=0
    start_time=time.time()
    for img,label in data_loader:
        optimizer.zero_grad()
        img=img.transpose(1,3).to(DEVICE)
        # print(img.shape)
        label=label.reshape(-1).to(DEVICE)
        outputs=model(img)

        loss=lossfunction(outputs,label)
        loss.backward()
        optimizer.step()
        scheduler.step()


        print(f'Epoch: {i + 1:02} | Each loss:{loss.item():.2f}  |  Each accuracy:{categorical_accuracy(outputs,label)*100:.2f}%')
        acc=acc+categorical_accuracy(outputs,label)
        each_loss=each_loss+loss.item()
    end_time = time.time()
    epoch_mins, epoch_secs = epoch_time(start_time, end_time)
    writer.add_scalar('Train/Loss',each_loss/len(data_loader),i)
    writer.add_scalar('Train/Accuracy',acc/len(data_loader),i)
    torch.save(model.state_dict(), str(i)+'CNN-fc-model.pt')
    print(f'Epoch: {i+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
    print(f'\tTrain Loss: {each_loss/len(data_loader):.3f} | Train Acc: {acc*100/len(data_loader):.2f}%')


# In[ ]:
































# In[ ]:




