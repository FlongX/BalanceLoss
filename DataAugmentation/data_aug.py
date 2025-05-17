import cv2
import torch
import random
import numpy as np
from scipy import ndimage
from scipy.ndimage import distance_transform_edt as distance
from scipy.ndimage import zoom, gaussian_filter, map_coordinates


def one_hot2dist(label):
    if label.sum == 0:
        d_map = np.zeros_like(label)
    else:
        posmask = label.astype(bool)
        negmask = ~posmask
        d_map = distance(negmask) * negmask - (distance(posmask) - 1) * posmask
    return d_map


def random_rot_flip(image, label):
    if random.random() < 0.25:
        k = np.random.randint(0, 4)
        image = np.rot90(image, k, axes=(1, 2))
        label = np.rot90(label, k)
        axis = np.random.randint(0, 2)
        image = np.flip(image, axis=axis + 1).copy()
        label = np.flip(label, axis=axis).copy()
    return image, label


def random_rotate(image, label):
    if random.random() < 0.25:
        angle = np.random.randint(-15, 15)
        image = ndimage.rotate(image, angle, axes=(1, 2), order=0, reshape=False)
        label = ndimage.rotate(label, angle, order=0, reshape=False)
    return image, label


def rewrap(coord, outsize):  # 防止超出边界
    return int(min(max(coord, 0), outsize))


def elastic_transform(image, label, alpha, sigma,
                      alpha_affine, random_state=None):
    if random.random() < 0.5:
        if random_state is None:
            random_state = np.random.RandomState(None)

        shape = image[0, :, :].shape
        shape_size = shape[:2]
        # Random affine
        center_square = np.float32(shape_size) // 2
        square_size = min(shape_size) // 3
        # pts1: 仿射变换前的点(3个点)
        pts1 = np.float32([center_square + square_size,
                           [center_square[0] + square_size,
                            center_square[1] - square_size],
                           center_square - square_size])
        # pts2: 仿射变换后的点
        pts2 = pts1 + random_state.uniform(-alpha_affine, alpha_affine,
                                           size=pts1.shape).astype(np.float32)
        # 仿射变换矩阵
        M = cv2.getAffineTransform(pts1, pts2)
        # 对image进行仿射变换.
        imageB1 = cv2.warpAffine(image[0, :, :], M, shape_size[::-1], borderMode=cv2.BORDER_REFLECT_101)
        imageB2 = cv2.warpAffine(image[1, :, :], M, shape_size[::-1], borderMode=cv2.BORDER_REFLECT_101)
        imageB3 = cv2.warpAffine(image[2, :, :], M, shape_size[::-1], borderMode=cv2.BORDER_REFLECT_101)
        labelB = cv2.warpAffine(label, M, shape_size[::-1], borderMode=cv2.BORDER_REFLECT_101)

        dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha
        dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha
        # generate meshgrid
        x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
        # x+dx,y+dy
        indices = np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1))
        # bilinear interpolation
        image1 = map_coordinates(imageB1, indices, order=1, mode='constant').reshape(shape)
        image2 = map_coordinates(imageB2, indices, order=1, mode='constant').reshape(shape)
        image3 = map_coordinates(imageB3, indices, order=1, mode='constant').reshape(shape)
        label = map_coordinates(labelB, indices, order=1, mode='constant').reshape(shape)
        image1 = np.expand_dims(image1, axis=0)
        image2 = np.expand_dims(image2, axis=0)
        image3 = np.expand_dims(image3, axis=0)
        image = np.concatenate((image1, image2, image3), axis=0)
    return image, label


class RandomGenerator(object):
    def __init__(self, output_size, split=None):
        self.output_size = output_size
        self.split = split

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        if self.split == 'train':
            image_aug, label_aug = random_rot_flip(image, label)
            image_aug, label_aug = random_rotate(image_aug, label_aug)
            alpha = random.uniform(200, 800)
            sigma = random.uniform(10, 20)
            image_aug, label_aug = elastic_transform(image_aug, label_aug, alpha, sigma, sigma)
            if label_aug.sum() == 0:
                image_aug, label_aug = image, label
            image = image_aug
            label = label_aug

        z, x, y = image.shape

        if x != self.output_size[0] or y != self.output_size[1]:
            image = zoom(image, (1, self.output_size[0] / x, self.output_size[1] / y), order=3)
            label = zoom(label, (self.output_size[0] / x, self.output_size[1] / y), order=0)

        foreground_num = torch.tensor(label.sum().astype(np.float32))
        background_num = torch.tensor(self.output_size[0] ** 2 - label.sum().astype(np.float32))
        d_map = one_hot2dist(label)

        image = torch.from_numpy(image)
        label = torch.from_numpy(label)

        image = image / 255

        sample = {'image': image, 'label': label.long(), 'f_n': foreground_num, 'b_n': background_num, 'd_map': d_map}

        return sample