


from torch.utils.data import random_split
import torch


import datasets

from torch import nn
from torchvision import datasets, transforms
from sklearn.model_selection import train_test_split

import kagglehub

# Download latest version
path = kagglehub.dataset_download("iamsouravbanerjee/animal-image-dataset-90-different-animals")

print("Path to dataset files:", path)

from torch.utils.data import DataLoader, random_split
IMG_SIZE = 128
transform = transforms.Compose([
    
    transforms.Resize(IMG_SIZE),
    transforms.CenterCrop(IMG_SIZE),
    # Convert images to PyTorch tensors and scale to [0, 1]
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

animal_dataset = datasets.ImageFolder(root=path,  # Specify the root directory of the dataset
                               transform=transform)

train_ratio = 0.8  # %80 Train, %20 Validation
train_size = int(train_ratio * len(animal_dataset))
val_size = len(animal_dataset) - train_size

train_dataset, val_dataset = random_split(animal_dataset, [train_size, val_size])

train_dataloaded=torch.utils.data.DataLoader(train_dataset,32,num_workers=2,shuffle=True,drop_last=True)
valid_dataloaded=torch.utils.data.DataLoader(val_dataset,32,num_workers=2,drop_last=True)
  # Apply the defined transformations to the dataset
