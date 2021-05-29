import numpy as np
import pathlib
import pandas as pd
from PIL import Image 
import json
import re

class ComplexMNISTDataset:
    def __init__(self, root, transform=None, dataset_type='train'):
        self.root = pathlib.Path(root)
        self.transform = transform
        self.dataset_type = dataset_type.lower()
        data_path = pathlib.Path(str(self.root / self.dataset_type / self.dataset_type) + '_labels_0.1.csv')
        self.data = pd.read_csv(data_path)

    def _getitem(self, index):
        row = self.data.iloc[index]
        img_path = self.root / '/'.join(row['Output File'].split('/')[1:])
        image = np.array(Image.open(img_path))
        # convert to 3 channels
        image = np.stack((image,)*3, axis=-1)
        composite = row['Composite class']
        component_idx = np.unique([int(x) for x in re.split(', |\s', row['Component classes'][1:-1])])
        component = np.zeros(10)
        component[component_idx] = 1

        if self.transform:
            image = self.transform(image)
        
        return image, composite, component

    def __getitem__(self, index):
        return self._getitem(index)

    def __len__(self):
        return len(self.data.index)

    def __repr__(self):
        return f'ComplexMNIST dataset - {self.dataset_type} with {self.__len__()} items'
