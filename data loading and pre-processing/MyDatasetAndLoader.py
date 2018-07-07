# coding: utf-8

import os
import numpy as np
import pandas as pd
import torch
from skimage import io
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

"""Dataset and DataLoader.

这里使用需要创建两个目录，images/和blendshapes/。
images中放置图片。
blendshapes中放置一个记录image对应blendshapes的csv文件，第一列是与image对应的timestamp，第二列之后是blendshapes.
"""

# generate dataset
class FaceDataset(Dataset):
    """Face Blendshapes datasets."""
    
    def __init__(self, images_dir, blendshapes_file, transform=None):
        """
        Args:
            images_dir (string): Directory with all images.
            blendshapes_file (string): The blendshapes file.
            transform (callable, optional): The optional transform to a sample.
        """
        self.images_dir = images_dir
        self.df_blendshapes = pd.read_csv(blendshapes_file, encoding='utf-8')
        self.transform = transform
    
    def __len__(self):
        return len(self.df_blendshapes)
    
    def __getitem__(self, idx):
        img_name = os.path.join(self.images_dir, '{}.png'.format(self.df_blendshapes.iloc[idx, 0]))

        image = io.imread(img_name).astype(np.float64)
        blendshape = self.df_blendshapes.iloc[idx, 1:]
        sample = {'image': image, 'blendshape': blendshape}
        
        if(self.transform):
            sample = self.transform(sample)
            
        return sample


# transform and ToTensor
class Norm(object):
    """Normalized the images and transpose """

    def __call__(self, sample):
        image, blendshape = sample['image'], sample['blendshape']
        norm_image = np.transpose(self.normalize(image), (2, 0, 1))
        return {'image': norm_image, 'blendshape': blendshape}
    
    def normalize(self, image):
        return (image - image.min()) / (image.max() - image.min())


class ToTensor(object):
    """Convert ndarray in sample to Tensor."""
    
    def __call__(self, sample):
        image, blendshape = sample['image'], sample['blendshape']
        return {'image': torch.from_numpy(image),
                'blendshape': torch.from_numpy(blendshape.values).double()}


if(__name__ == '__main__'):
    face_dataset = FaceDataset(images_dir='./images/',
                               blendshapes_file='./blendshapes/blendshapes.csv',
                               transform=transforms.Compose([
                                   Norm(),
                                   ToTensor()
                               ]))
    dataloader = DataLoader(face_dataset, batch_size=4, shuffle=True, num_workers=4)
    for step, batch_sample in enumerate(dataloader):
        print(batch_sample['image'].shape, batch_sample['blendshape'][:, 0].numpy().tolist())

