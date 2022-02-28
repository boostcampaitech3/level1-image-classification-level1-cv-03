import torch
from torchvision import transforms
from torch.utils.data import Dataset



class MaskDataset(Dataset):

    def __init__(self, input_data, output_data, transform):
        self.X = input_data
        self.Y = output_data
        self.transform = transform
        
    def __len__(self):
        return len(self.X)
        
    def __getitem__(self, index):
        image = self.transform(self.X[index])
        label = self.Y[index]
        return image, label
