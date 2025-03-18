import os
import cv2
import numpy as np

from torch.utils.data import Dataset

class HouseFacades(Dataset):
    def __init__(self, root_path, split, transform):
        assert split in ['test', 'train', 'val'], \
            f"Unknown split {split}."
        
        self.__image_folder = os.path.join(root_path, split)
        self.__images = os.listdir(self.__image_folder)
        self.__transform = transform


    def __len__(self) -> int:
        return len(self.__images)
    
    def __getitem__(self, index) -> tuple[np.ndarray, np.ndarray]:
        image_path = os.path.join(self.__image_folder, self.__images[index])
        img = cv2.imread(image_path)

        height, width = img.shape

        image = img[:, :(width//2)]
        mask = img[:, (width//2):]

        if self.__transform is not None:
            image, mask = self.__transform(image, mask)

        return image, mask
