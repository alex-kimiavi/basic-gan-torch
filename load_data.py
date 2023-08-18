import torch 
import torchvision.transforms as T
import os

from PIL import Image

"""Be sure to download the CelebA dataset at the following link"""
"""https://www.kaggle.com/datasets/jessicali9530/celeba-dataset"""
class CelebADataset(torch.utils.data.Dataset):
    
    def __init__(self, dataset_dir = '../dataset/img_align_celeba/img_align_celeba/', img_size = (64, 64)):
        self.img_paths = [f'{dataset_dir}/{f}' for f in os.listdir(dataset_dir)]
        self.img_size = img_size
        self.transform = T.Compose([T.ToTensor(), T.Resize((96, 96)), T.CenterCrop((64, 64)), T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img = Image.open(self.img_paths[idx])
        return self.transform(img), []  # image, label

if __name__ == '__main__':
    dset = CelebADataset()