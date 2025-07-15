import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from sklearn.preprocessing import LabelEncoder

from utils import resize_image, flip_image, normalize
from config import Config

class OAIData:
    def __init__(self, phase, transform=None):
        self.phase = phase
        self.transform = transform
        self.zero_img = np.zeros(Config.IMAGE_SIZE)
        
        # Load the CSV with names and labels
        self.df = pd.read_csv(Config.get_data_file_path(phase))
        
        # Load image data if needed
        self.images = {}
        self.image_names = {}

        #npz file for knee images  
        npzfile = np.load(Config.get_knee_data_path())
        self.images['knee'] = npzfile['x']
        self.image_names['knee'] = list(npzfile['y'])
        
        # Preprocess all data
        self.samples = []
        self.prepare_samples()

    def prepare_samples(self):
        for _, row in self.df.iterrows():
            name = row['name']
            label = row['WOMAC']
            
            sample = {
                'name': name,
                'label': label,
                'img': self.get_image(name) if 'knee' in self.images else None,
                'img_mask': 1 if 'knee' in self.images and name in self.image_names['knee'] else 0
            }
            self.samples.append(sample)

    def get_image(self, name):
        if name in self.image_names['knee']:
            idx = self.image_names['knee'].index(name)
            img = self.images['knee'][idx]

            flip_val = False
            if 'RIGHT' in name:
                flip_val = True

            img = self.preprocess_img(img, flip=flip_val)
            return img
        return self.zero_img

    def preprocess_img(self, img, flip=False):
        img = resize_image(img.astype(np.float32), 
                         target_size=list(Config.IMAGE_SIZE))            
        img = normalize(img, percentage_clip=Config.PERCENTAGE_CLIP, zero_centered=Config.ZERO_CENTERED)      
        if flip:
            img = flip_image(img)  
        return img

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = {
            'name': self.samples[idx]['name'],
            'label': self.samples[idx]['label'],
            'img': self.samples[idx]['img'].copy() if self.samples[idx]['img'] is not None else None,
            'img_mask': self.samples[idx]['img_mask']
        }

        
        if sample['img'] is not None and self.transform is not None:
            

            sample['img'] = self.transform(sample['img'])
        
        return sample

class ToTensorWithChannel(object):
    def __call__(self, sample):
        sample_tensor = torch.from_numpy(sample.copy())
        sample_tensor = sample_tensor.repeat(Config.NUM_CHANNELS, 1, 1)
        return sample_tensor

def get_loader(phase, shuffle, drop_last=False):
    transform = transforms.Compose([ToTensorWithChannel()])
    
    dataset = OAIData(phase=phase, transform=transform)
    
    loader = DataLoader(
        dataset, 
        batch_size=Config.BATCH_SIZE, 
        shuffle=shuffle, 
        drop_last=drop_last,
        num_workers=Config.NUM_WORKERS
    )

    
    print(f"Finished loading {len(dataset)} {phase} samples")
    return loader