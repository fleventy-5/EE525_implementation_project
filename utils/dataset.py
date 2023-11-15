import os
from torch.utils.data import DataLoader,Dataset
from PIL import Image
from torchvision.transforms import v2
import numpy as np

class CustomImageDataset(Dataset):
    def __init__(self, train_dir, label_dir, transform=None):
        self.train_dir = train_dir
        self.label_dir = label_dir
        self.transform = transform
        self.train_image_paths = [os.path.join(train_dir, filename) for filename in os.listdir(train_dir)]
        self.label_image_paths = [os.path.join(label_dir, filename) for filename in os.listdir(label_dir)]

    def __len__(self):
        return len(self.train_image_paths)

    def __getitem__(self, idx):
        train_image_path = self.train_image_paths[idx]
        label_image_path = self.label_image_paths[idx]
        
        train_image = Image.open(train_image_path)
        label_image = Image.open(label_image_path)
        # train_image = (np.asarray(train_image)/255.0) 
        # label_image = (np.asarray(label_image)/255.0) 

        if self.transform:
            train_image = self.transform(train_image)
            label_image = self.transform(label_image)
            

        return train_image, label_image

def load_data(train_path, label_path):
    data_transforms = v2.Compose([
        v2.Resize(size=(64,64)),
        #v2.RandomCrop(size=(32, 32)),
        # v2.Resize(size=(64,64)),
        v2.ToTensor()
    ])

    custom_dataset = CustomImageDataset(
        train_dir=train_path,
        label_dir=label_path,
        transform=data_transforms
    )

    data_loader = DataLoader(
        dataset=custom_dataset,
        batch_size=1,
        num_workers=1,
        shuffle=True
    )
    return data_loader