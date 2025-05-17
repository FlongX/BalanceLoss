import os
import cv2 as cv
from torch.utils.data import Dataset


class BUSIDataset(Dataset):
    def __init__(self, base_dir, split, outsize, transform=None):
        self.transform = transform
        self.split = split
        self.image_list = []
        self.label_list = []
        self.outsize = outsize

        if self.split == 'train':
            for i in range(1, 281):
                case_image_path = os.path.join(base_dir, 'benign', f"{i:04d}" + 'image.png')
                case_label_path = os.path.join(base_dir, 'benign', f"{i:04d}" + 'mask.png')

                self.image_list.append(case_image_path)
                self.label_list.append(case_label_path)
            for i in range(1, 136):

                case_image_path = os.path.join(base_dir, 'malignant', f"{i:04d}" + 'image.png')
                case_label_path = os.path.join(base_dir, 'malignant', f"{i:04d}" + 'mask.png')

                self.image_list.append(case_image_path)
                self.label_list.append(case_label_path)

        if self.split == 'val':
            for i in range(281, 351):
                case_image_path = os.path.join(base_dir, 'benign', f"{i:04d}" + 'image.png')
                case_label_path = os.path.join(base_dir, 'benign', f"{i:04d}" + 'mask.png')

                self.image_list.append(case_image_path)
                self.label_list.append(case_label_path)

            for i in range(136, 170):
                case_image_path = os.path.join(base_dir, 'malignant', f"{i:04d}" + 'image.png')
                case_label_path = os.path.join(base_dir, 'malignant', f"{i:04d}" + 'mask.png')

                self.image_list.append(case_image_path)
                self.label_list.append(case_label_path)

        if self.split == 'test':
            for i in range(351, 438):
                case_image_path = os.path.join(base_dir, 'benign', f"{i:04d}" + 'image.png')
                case_label_path = os.path.join(base_dir, 'benign', f"{i:04d}" + 'mask.png')

                self.image_list.append(case_image_path)
                self.label_list.append(case_label_path)

            for i in range(170, 211):
                case_image_path = os.path.join(base_dir, 'malignant', f"{i:04d}" + 'image.png')
                case_label_path = os.path.join(base_dir, 'malignant', f"{i:04d}" + 'mask.png')

                self.image_list.append(case_image_path)
                self.label_list.append(case_label_path)

        print(self.image_list[0])
        print('image 长度：', len(self.image_list), 'label 长度： ', len(self.label_list))
        print(self.label_list[0])

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