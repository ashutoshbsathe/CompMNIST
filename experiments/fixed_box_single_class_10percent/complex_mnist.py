import numpy as np
import pathlib
import pandas as pd
from PIL import Image 
import json

class ComplexMNISTDataset:
    def __init__(self, root, transform=None, dataset_type='train'):
        self.root = pathlib.Path(root)
        self.transform = transform
        self.dataset_type = dataset_type.lower()
        data_path = pathlib.Path(str(self.root / self.dataset_type / self.dataset_type) + '_labels_0.1_only3.csv')
        self.data = pd.read_csv(data_path)

    def _getitem(self, index):
        row = self.data.iloc[index]
        img_path = self.root / '/'.join(row['Output File'].split('/')[1:])
        image = np.array(Image.open(img_path))
        # convert to 3 channels
        image = np.stack((image,)*3, axis=-1)
        gt_path = pathlib.Path(str(img_path) + '_gt.json')
        with open(gt_path, 'r') as f:
            groundtruth = json.load(f)
        composite_class = groundtruth['composite_class']
        component_boxes = []
        component_labels = []
        for info in groundtruth['component_classes']:
            component_boxes.append(np.array([info['x'], info['y'], info['x'] + info['width'], info['y'] + info['height']]))
            component_labels.append(info['class'])
        if len(component_boxes) < 32:
            # we may sample less points due to disconnected paths
            duplicate_idx = 32 - len(component_boxes)
            for info in groundtruth['component_classes'][-duplicate_idx:]:
                component_boxes.append(np.array([info['x'], info['y'], info['x'] + info['width'], info['y'] + info['height']]))
                component_labels.append(info['class'])
        component_boxes = np.array(component_boxes, dtype=np.float64)
        component_labels = np.array(component_labels, dtype=np.float64)
        assert np.unique(component_labels).shape[0] == 1, f'np.unique(component_labels) = {np.unique(component_labels)}'
        if self.transform:
            image = self.transform(image)
        assert len(component_boxes) == len(component_labels)
        assert len(component_labels) == 32, f'Found length = {len(component_labels)} boxes shape = {component_boxes.shape} labels shape = {component_labels.shape}'
        component_boxes = component_boxes.reshape(128)
        return index, image, composite_class, component_boxes, component_labels

    def __getitem__(self, index):
        image_id, image, composite_class, component_boxes, component_labels = self._getitem(index)
        return image, composite_class, component_boxes, int(np.unique(component_labels)[0])

    def __len__(self):
        return len(self.data.index)

    def __repr__(self):
        return f'ComplexMNIST dataset - {self.dataset_type} with {self.__len__()} items'
