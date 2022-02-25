import os
import pandas as pd
from PIL import Image
import torch
train_images_path = '/opt/ml/input/data/train/images'
train_dir = os.listdir(train_images_path)

train_dir.sort()
train_dir = train_dir[2700:]

images_address_list = [train_images_path + "/" + i + "/" + j for i in train_dir for j in os.listdir(train_images_path + "/" + i) if '._' not in j]

images_list = [Image.open(i) for i in images_address_list]

gender = [i.split('/')[-2].split('_')[-3] for i in images_address_list]
age = [int(i.split('/')[-2].split('_')[-1]) for i in images_address_list]
ismask = [i.split('/')[-1].split('.')[0] for i in images_address_list]

for i in range(len(ismask)):
    if '_mask' in ismask[i]:
        ismask[i] = '0'

for i in range(len(ismask)):
    if 'mask' in ismask[i]:
        ismask[i] = '1'
    elif 'normal' in ismask[i]:
        ismask[i] = '2'

import numpy as np

ismask = list(map(int,np.array(ismask)))
    
    # 0 : incorrect  
    # 1 : wear  
    # 2 : normal  
    
for i in range(len(gender)):
    if gender[i] =='male':
        gender[i] = 0
    elif gender[i] =='female':
        gender[i] = 1
    
    # male : 0  
    # female : 1

Label_class = []
for i,j,k in zip(ismask,gender,age):
    Label_class.append([i,j,k])
    
class_description = []
for i in Label_class:
    ismask,gender,age = i
    if ismask==1 and gender==0 and age<30:
        class_description.append(0)
    elif ismask==1 and gender==0 and age>=30 and age<60:
        class_description.append(1)
    elif ismask==1 and gender==0 and age>=60:
        class_description.append(2)
    elif ismask==1 and gender==1 and age<30:
        class_description.append(3)
    elif ismask==1 and gender==1 and age>=30 and age<60:
        class_description.append(4)
    elif ismask==1 and gender==1 and age>=60:
        class_description.append(5)
    elif ismask==0 and gender==0 and age<30:
        class_description.append(6)
    elif ismask==0 and gender==0 and age>=30 and age<60:
        class_description.append(7)
    elif ismask==0 and gender==0 and age>=60:
        class_description.append(8)
    elif ismask==0 and gender==1 and age<30:
        class_description.append(9)
    elif ismask==0 and gender==1 and age>=30 and age<60:
        class_description.append(10)
    elif ismask==0 and gender==1 and age>=60:
        class_description.append(11)
    elif ismask==2 and gender==0 and age<30:
        class_description.append(12)
    elif ismask==2 and gender==0 and age>=30 and age<60:
        class_description.append(13)
    elif ismask==2 and gender==0 and age>=60:
        class_description.append(14)
    elif ismask==2 and gender==1 and age<30:
        class_description.append(15)
    elif ismask==2 and gender==1 and age>=30 and age<60:
        class_description.append(16)
    elif ismask==2 and gender==1 and age>=60:
        class_description.append(17)

class_description = torch.LongTensor(class_description)

# input : images_list / image_data in list
# output : class_description / integer_data in list

class Data():
    def __init__(self):
        self.images_list = images_list
        self.class_description = class_description



if __name__ == "__main__":
    
    print(class_description)   

    