import torch
import cv2
import os

class SegmentationDataset(torch.utils.data.Dataset):
    def __init__(self, path_to_dataset, instance_names_list, transforms):
        '''
        path_to_dataset - путь до корневой папки с датасетом
        instance_names_list - список имен экземпляров БЕЗ РАСШИРЕНИЯ!
        transforms - аугментация изображений
        '''
        super().__init__()
        self.path_to_dataset = path_to_dataset 
        
        self.instance_names_list = instance_names_list
        self.transforms = transforms

    def __len__(self):
        return len(self.instance_names_list)

    def __getitem__(self, idx):
        instance_name = self.instance_names_list[idx]
        path_to_image = os.path.join(self.path_to_dataset, 'images', instance_name + '.jpg')
        path_to_label = os.path.join(self.path_to_dataset, 'labels', instance_name + '.png')

        image = cv2.imread(path_to_image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # метки читаем как одноканальное изображение
        label = cv2.imread(path_to_label, cv2.IMREAD_UNCHANGED)/255

        #print(label.dtype)

        # Тут используется библиотека Albumentations
        transformed = self.transforms(image=image, mask=label)
        image = transformed['image']
        # почему-то albumentations приводит маску к типу int32
        # поэтому, требуется явное приведение типа к int64
        label = transformed['mask'].long() 
        #label = transformed['mask'].unsqueeze(0).float()
        #print(label.dtype)

        return image, label
