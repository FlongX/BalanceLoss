import os
import cv2 as cv
from torch.utils.data import Dataset
from scipy.ndimage.interpolation import zoom


class CVCDataset(Dataset):
    def __init__(self, base_dir, split, outsize, transform=None):
        self.output_size = None
        self.transform = transform
        self.split = split
        self.image_list = []
        self.label_list = []
        self.outsize = outsize

        image_dir = os.path.join(base_dir, 'image')
        label_dir = os.path.join(base_dir, 'label')

        num_dir = os.listdir(image_dir)
        if self.split == 'train':
            for png in num_dir:
                if int(png.split('.')[0]) <= 392:
                    image_path = os.path.join(image_dir, png)
                    label_path = os.path.join(label_dir, png)

                    label_data = cv.imread(label_path)
                    label_data = label_data[:, :, 0]
                    label_data[label_data < 125] = 0
                    label_data[label_data > 0] = 1
                    x, y = label_data.shape

                    if x != 288:
                        print(image_path)
                    label_data = zoom(label_data, (self.outsize[0] / x, self.outsize[1] / y), order=0)
                    if label_data.sum() < 10:
                        continue

                    self.image_list.append(image_path)
                    self.label_list.append(label_path)
        if self.split == 'val':
            for png in num_dir:
                if 392 < int(png.split('.')[0])<=490:
                    image_path = os.path.join(image_dir, png)
                    label_path = os.path.join(label_dir, png)

                    self.image_list.append(image_path)
                    self.label_list.append(label_path)
        if self.split == 'test':
            for png in num_dir:
                if 490 < int(png.split('.')[0]):
                    image_path = os.path.join(image_dir, png)
                    label_path = os.path.join(label_dir, png)

                    self.image_list.append(image_path)
                    self.label_list.append(label_path)

        print('image 长度：', len(self.image_list), 'label 长度： ', len(self.label_list))

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):

        image_data_path = os.path.join(self.image_list[idx])
        label_data_path = os.path.join(self.label_list[idx])

        label_data = cv.imread(label_data_path)
        label_data = label_data[:, :, 0]
        label_data[label_data < 125] = 0
        label_data[label_data > 0] = 1

        image_data = cv.imread(image_data_path)
        image_data = image_data.transpose([2, 0, 1])  # c, h, w

        image, label = image_data, label_data

        sample = {'image': image, 'label': label}
        if self.transform:
            sample = self.transform(sample)

        return sample
