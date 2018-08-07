from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy

'''
    아래 이미지는 이미지를 담고있는 [메인 폴더] 명 안에 [train], [val]의 폴더가 있고, 
        각각의 폴더 안에 train, val의 이미지가 있는 조건 하에 이미지를 불러오는 코드이다.
    
    ----------- 구조 -----------
    [메인 폴더]
    
        -> [train]
            -> 사진
            
        -> [val]
            -> 사진
    ---------------------------
'''

# data_transforms는 이미지 형식의 틀을 만들어 주는 것.(이미지를 불러오는 단계가 아니다.)
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

# data_dir는 이미지가 담겨있는 폴더 명 이다.
data_dir = 'hymenoptera_data'

# image_datasets에서 이미지의 path를 담는다.  (image_datasets['train'], image_datasets['val']이 생성된다.)
#                                       ( 각각 hymenoptera_data/train, hymenoptera_data/val 의 사전값을 갖는다)
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                          data_transforms[x])
                  for x in ['train', 'val']}

# datalodaers에 이미지를 넣는다. 조건으로 4장씩 넣으며(배치사이즈=4), 랜덤한 이미지로 넣는다. (num_worksers는 ??)
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4,
                                             shuffle=True, num_workers=4)
              for x in ['train', 'val']}

# dataset_sizes에 각각(train폴더, val폴더)의 이미지 개수를 저장
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}

# class_names에 클래스 이름 저장(ants, bees)
#   image_datasets['train'].classes[0] = ants,
#   image_datasets['train'].classes[1] = bees,
#   image_datasets['val'].classes[0] = ants,
#   image_datasets['val'].classes[1] = bees
class_names = image_datasets['train'].classes

# inputs에 이미지를 넣고, classes에 클래스를 넣는 과정
inputs, classes = next(iter(dataloaders['train']))  # inputs.size는 4, 3, 224, 224가 잡히는데, 4는 배치사이즈(이미지4장), 3은 채널, 224는

aaa = 3
