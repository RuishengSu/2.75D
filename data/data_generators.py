import math
import re
from scipy.ndimage import rotate
from scipy.ndimage.interpolation import shift, zoom

import sys
# print(sys.path)
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
# print(f"Current Directory: {current_dir}")
# Add custom module paths
sys.path.append(current_dir)
sys.path.append(parent_dir)
from project_config import *


import os
import cv2
import argparse
import torch
import random
import numpy as np
import pandas as pd
import json
import datetime
import pydicom
from sklearn.utils import shuffle
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from monai.transforms import AddChannel, Compose, RandAffine, RandZoom, RandRotate, RandFlip, apply_transform, ToTensor
import torchvision.utils as vutils

# image transform
import random
import numbers
from PIL import Image, ImageFilter
from torchvision import transforms
from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage.interpolation import map_coordinates
import matplotlib.pyplot as plt


def get_labels(filenames):
    y = np.empty(len(filenames), dtype=int)
    seriesuids = []
    coordXs = np.empty(len(filenames), dtype=np.float32)
    coordYs = np.empty(len(filenames), dtype=np.float32)
    coordZs = np.empty(len(filenames), dtype=np.float32)
    for i, ID in enumerate(filenames):
        # seriesuids = np.append(seriesuids, re.search(r'subset[0-9]/(.*?).mhd', ID).group(1))
        # coordsXYZ = re.search('pos(.*?)_[0-9]+_(fake|real)_size64x64.npy', ID).group(1)
        result1 = re.search(r'subset[0-9]/(.*?)_pos', ID)
        if result1 != None:
            print(result1.group(1))
        else:
            print('None')
        seriesuids = np.append(seriesuids, result1)
        #seriesuids = np.append(seriesuids, re.search(r'subset[0-9]/(.*?)_pos', ID).group(1))

        """coordsXYZ = re.search('pos(.*?)_(fake|real)_size64x64.npy', ID)
        if coordsXYZ != None:
            print(coordsXYZ.group(1))
        else:
            print('None')"""
        coordsXYZ = re.search('pos(.*?)_(fake|real)_size64x64.npy', ID).group(1)


        coordXs[i] = float(coordsXYZ.split("_")[0])
        coordYs[i] = float(coordsXYZ.split("_")[1])
        coordZs[i] = float(coordsXYZ.split("_")[2])
        if 'real_' in ID:
            y[i] = 1
        elif 'fake_' in ID:
            y[i] = 0
        else:
            raise Exception('Filename does not contain expected keywords. Label was not assigned properly')
    return y, seriesuids, coordXs, coordYs, coordZs


def get_slices_from_blob(blob, blob_size):
    number_of_planes = 9
    extracted_planes = np.zeros((blob_size, blob_size, number_of_planes), dtype=blob.dtype)

    # plane 1
    extracted_planes[:, :, 0] = (blob[:, :, blob_size // 2] + blob[:, :, blob_size // 2 + 1]) / 2
    # plane 2
    extracted_planes[:, :, 1] = (blob[:, blob_size // 2, :] + blob[:, blob_size // 2 + 1, :]) / 2
    # plane 3
    extracted_planes[:, :, 2] = (blob[blob_size // 2, :, :] + blob[blob_size // 2 + 1, :, :]) / 2

    # plane 4
    extracted_planes[:, :, 3] = np.diagonal(blob[:, :, :], 0, 0, 1)
    # plane 5
    extracted_planes[:, :, 4] = np.diagonal(blob[:, :, :], 0, 0, 2)
    # plane 6
    extracted_planes[:, :, 5] = np.diagonal(blob[:, :, :], 0, 1, 2)

    # plane 7
    extracted_planes[:, :, 6] = np.diagonal(np.rot90(blob[:, :, :], k=1, axes=(0, 1)), 0, 0, 1)
    # plane 8
    extracted_planes[:, :, 7] = np.diagonal(np.rot90(blob[:, :, :], k=1, axes=(0, 2)), 0, 0, 2)
    # plane 9
    extracted_planes[:, :, 8] = np.diagonal(np.rot90(blob[:, :, :], k=1, axes=(1, 2)), 0, 1, 2)

    return extracted_planes


def convert_to_euclidean_space(blob_radius, inner_radius, alpha,
                               beta):  # Z is the height; space center is the top left corner of the cubic
    verticle_angle = (math.pi / 2.0) - beta
    inner_radius_XY = inner_radius * math.cos(verticle_angle)
    Z = int(round(blob_radius - inner_radius * math.sin(verticle_angle)))
    X = int(round(blob_radius + inner_radius_XY * math.cos(alpha)))
    Y = int(round(blob_radius + inner_radius_XY * math.sin(alpha)))
    return X, Y, Z


def get_angels(N):
    sphere_step_size = math.pi / float(N)
    # nr_of_points_on_each_circle = 2*N
    alpha_list = []  # horizontal angles, 0-2pi
    beta_list = []  # vitical angles, 0-pi
    alpha_list.append(0)
    beta_list.append(0)

    vertical_beta_step = math.pi / N
    for i in range(0, int(math.ceil(N / 2.0))):
        vertical_beta_on_Ith_circle = (i + 1) * vertical_beta_step
        cos_angel = math.fabs(math.pi / 2 - vertical_beta_on_Ith_circle)
        perimeter = 2 * math.pi * math.cos(cos_angel)
        nr_of_points_on_circle = int(round(perimeter / sphere_step_size))
        for j in range(0, nr_of_points_on_circle):
            alpha_step_size = 2 * math.pi / float(nr_of_points_on_circle)
            beta_step_size_between_circles = vertical_beta_step / float(nr_of_points_on_circle)
            alpha_list.append((j + 1) * alpha_step_size)
            beta_list.append(i * vertical_beta_step + (j + 1) * beta_step_size_between_circles)
    for i in range(int(math.ceil(N / 2.0)), N):
        vertical_beta_on_Ith_circle = i * vertical_beta_step
        cos_angel = math.fabs(math.pi / 2 - vertical_beta_on_Ith_circle)
        perimeter = 2 * math.pi * math.cos(cos_angel)
        nr_of_points_on_circle = int(round(perimeter / sphere_step_size))
        for j in range(0, nr_of_points_on_circle):
            alpha_step_size = 2 * math.pi / float(nr_of_points_on_circle)
            beta_step_size_between_circles = vertical_beta_step / float(nr_of_points_on_circle)
            alpha_list.append((j + 1) * alpha_step_size)
            beta_list.append(i * vertical_beta_step + (j + 1) * beta_step_size_between_circles)
    return alpha_list, beta_list


def get_spiral_image(blob, spiral_image_height, augmentation=False, spiral_augmentation='none'):
    blob_size = blob.shape

    if blob_size[0] != blob_size[1] or blob_size[0] != blob_size[2]:
        raise ValueError("input blob is not a cubic!!")

    # Input parameters
    blob_radius = (blob_size[0] - 1) / 2.0
    nr_of_points_on_radius = spiral_image_height  # including the point on the out most sphere 32
    N = 9  # Divide a half circle pi into N parts, meaning the number of horizontal circles, including the other polar point

    alpha_list, beta_list = get_angels(N)
    radius_list = []
    radius_step = blob_radius / float(
        nr_of_points_on_radius)  # number of points on radius should be smaller than radius, since the points will be integer values
    for i in range(nr_of_points_on_radius, 0, -1):
        radius_list.append(i * radius_step)

    spiral_image = np.empty([radius_list.__len__(), alpha_list.__len__(), 1])

    if spiral_augmentation == '18Aug':
        select = random.randint(1, 18)
    elif spiral_augmentation == '6Aug':
        select = random.randint(1, 6)

    for row_index in range(0, radius_list.__len__()):
        radius_for_the_row = radius_list[row_index]
        for column_index in range(0, alpha_list.__len__()):
            current_alpha = alpha_list[column_index]
            current_beta = beta_list[column_index]
            X, Y, Z = convert_to_euclidean_space(blob_radius, radius_for_the_row, current_alpha, current_beta)

            if not augmentation or spiral_augmentation == 'none':
                spiral_image[row_index][column_index][0] = blob[X][Y][Z]
            else:
                if spiral_augmentation == '18Aug':
                    if select == 1:
                        spiral_image[row_index][column_index][0] = blob[X][Y][Z]
                    elif select == 2:
                        spiral_image[row_index][column_index][0] = blob[Y][Z][X]
                    elif select == 3:
                        spiral_image[row_index][column_index][0] = blob[Z][X][Y]
                    elif select == 4:
                        spiral_image[row_index][column_index][0] = blob[X][Z][Y]
                    elif select == 5:
                        spiral_image[row_index][column_index][0] = blob[Y][X][Z]
                    elif select == 6:
                        spiral_image[row_index][column_index][0] = blob[Z][Y][X]
                    elif select == 7:
                        spiral_image[row_index][column_index][0] = blob[blob_size[0] - 1 - X][Y][Z]
                    elif select == 8:
                        spiral_image[row_index][column_index][0] = blob[blob_size[0] - 1 - Y][Z][X]
                    elif select == 9:
                        spiral_image[row_index][column_index][0] = blob[blob_size[0] - 1 - Z][X][Y]
                    elif select == 10:
                        spiral_image[row_index][column_index][0] = blob[blob_size[0] - 1 - X][Z][Y]
                    elif select == 11:
                        spiral_image[row_index][column_index][0] = blob[blob_size[0] - 1 - Y][X][Z]
                    elif select == 12:
                        spiral_image[row_index][column_index][0] = blob[blob_size[0] - 1 - Z][Y][X]
                    elif select == 13:
                        spiral_image[row_index][column_index][0] = blob[blob_size[0] - 1 - X][blob_size[0] - 1 - Y][
                            blob_size[0] - 1 - Z]
                    elif select == 14:
                        spiral_image[row_index][column_index][0] = blob[blob_size[0] - 1 - Y][blob_size[0] - 1 - Z][
                            blob_size[0] - 1 - X]
                    elif select == 15:
                        spiral_image[row_index][column_index][0] = blob[blob_size[0] - 1 - Z][blob_size[0] - 1 - X][
                            blob_size[0] - 1 - Y]
                    elif select == 16:
                        spiral_image[row_index][column_index][0] = blob[blob_size[0] - 1 - X][blob_size[0] - 1 - Z][
                            blob_size[0] - 1 - Y]
                    elif select == 17:
                        spiral_image[row_index][column_index][0] = blob[blob_size[0] - 1 - Y][blob_size[0] - 1 - X][
                            blob_size[0] - 1 - Z]
                    elif select == 18:
                        spiral_image[row_index][column_index][0] = blob[blob_size[0] - 1 - Z][blob_size[0] - 1 - Y][
                            blob_size[0] - 1 - X]
                    else:
                        print(select)
                        print('wrong')
                elif spiral_augmentation == '6Aug':
                    if select == 1:
                        spiral_image[row_index][column_index][0] = blob[X][Y][Z]
                    elif select == 2:
                        spiral_image[row_index][column_index][0] = blob[X][Z][Y]
                    elif select == 3:
                        spiral_image[row_index][column_index][0] = blob[blob_size[0] - 1 - X][Y][Z]
                    elif select == 4:
                        spiral_image[row_index][column_index][0] = blob[blob_size[0] - 1 - X][Z][Y]
                    elif select == 5:
                        spiral_image[row_index][column_index][0] = blob[blob_size[0] - 1 - X][blob_size[0] - 1 - Y][
                            blob_size[0] - 1 - Z]
                    elif select == 6:
                        spiral_image[row_index][column_index][0] = blob[blob_size[0] - 1 - X][blob_size[0] - 1 - Z][
                            blob_size[0] - 1 - Y]
                    else:
                        print(select)
                        print('wrong')
                else:
                    print('wrong')

    return spiral_image


def get_spiral_image_3channels(blob, spiral_image_height, augmentation=False, spiral_augmentation='none'):
    blob_size = blob.shape

    if blob_size[0] != blob_size[1] or blob_size[0] != blob_size[2]:
        raise ValueError("input blob is not a cubic!!")

    # Input parameters
    blob_radius = (blob_size[0] - 1) / 2.0
    nr_of_points_on_radius = spiral_image_height  # including the point on the out most sphere
    N = 9  # Divide a half circle pi into N parts, meaning the number of horizontal circles, including the other polar point

    alpha_list, beta_list = get_angels(N)
    radius_list = []
    radius_step = blob_radius / float(
        nr_of_points_on_radius)  # number of points on radius should be smaller than radius, since the points will be integer values
    for i in range(nr_of_points_on_radius, 0, -1):
        radius_list.append(i * radius_step)

    spiral_image_3channels = np.empty([radius_list.__len__(), alpha_list.__len__(), 3])


    if spiral_augmentation == '18Aug':
        select0 = random.randint(1, 18)
        select1 = random.randint(1, 18)
        select2 = random.randint(1, 18)
        select = [select0, select1, select2]
    elif spiral_augmentation == '6Aug':
        select0 = random.randint(1, 6)
        select1 = random.randint(1, 6)
        select2 = random.randint(1, 6)
        select = [select0, select1, select2]



    for row_index in range(0, radius_list.__len__()):
        radius_for_the_row = radius_list[row_index]
        for column_index in range(0, alpha_list.__len__()):
            current_alpha = alpha_list[column_index]
            current_beta = beta_list[column_index]
            X, Y, Z = convert_to_euclidean_space(blob_radius, radius_for_the_row, current_alpha, current_beta)
            # old 3 channels
            # spiral_image_3channels[row_index][column_index][0] = blob[X][Y][Z]
            # spiral_image_3channels[row_index][column_index][1] = blob[X][Z][Y]
            # spiral_image_3channels[row_index][column_index][2] = blob[Z][X][Y]
            if not augmentation or spiral_augmentation == 'none':
                spiral_image_3channels[row_index][column_index][0] = blob[X][Y][Z]
                spiral_image_3channels[row_index][column_index][1] = blob[Y][Z][X]
                spiral_image_3channels[row_index][column_index][2] = blob[Z][X][Y]
            else:
                if spiral_augmentation == '18Aug':
                    for channel in range(0, 3):
                        if select[channel] == 1:
                            spiral_image_3channels[row_index][column_index][channel] = blob[X][Y][Z]
                        elif select[channel] == 2:
                            spiral_image_3channels[row_index][column_index][channel] = blob[Y][Z][X]
                        elif select[channel] == 3:
                            spiral_image_3channels[row_index][column_index][channel] = blob[Z][X][Y]
                        elif select[channel] == 4:
                            spiral_image_3channels[row_index][column_index][channel] = blob[X][Z][Y]
                        elif select[channel] == 5:
                            spiral_image_3channels[row_index][column_index][channel] = blob[Y][X][Z]
                        elif select[channel] == 6:
                            spiral_image_3channels[row_index][column_index][channel] = blob[Z][Y][X]
                        elif select[channel] == 7:
                            spiral_image_3channels[row_index][column_index][channel] = blob[blob_size[0] - 1 - X][Y][Z]
                        elif select[channel] == 8:
                            spiral_image_3channels[row_index][column_index][channel] = blob[blob_size[0] - 1 - Y][Z][X]
                        elif select[channel] == 9:
                            spiral_image_3channels[row_index][column_index][channel] = blob[blob_size[0] - 1 - Z][X][Y]
                        elif select[channel] == 10:
                            spiral_image_3channels[row_index][column_index][channel] = blob[blob_size[0] - 1 - X][Z][Y]
                        elif select[channel] == 11:
                            spiral_image_3channels[row_index][column_index][channel] = blob[blob_size[0] - 1 - Y][X][Z]
                        elif select[channel] == 12:
                            spiral_image_3channels[row_index][column_index][channel] = blob[blob_size[0] - 1 - Z][Y][X]
                        elif select[channel] == 13:
                            spiral_image_3channels[row_index][column_index][channel] = blob[blob_size[0] - 1 - X][blob_size[0] - 1 - Y][
                                blob_size[0] - 1 - Z]
                        elif select[channel] == 14:
                            spiral_image_3channels[row_index][column_index][channel] = blob[blob_size[0] - 1 - Y][blob_size[0] - 1 - Z][
                                blob_size[0] - 1 - X]
                        elif select[channel] == 15:
                            spiral_image_3channels[row_index][column_index][channel] = blob[blob_size[0] - 1 - Z][blob_size[0] - 1 - X][
                                blob_size[0] - 1 - Y]
                        elif select[channel] == 16:
                            spiral_image_3channels[row_index][column_index][channel] = blob[blob_size[0] - 1 - X][blob_size[0] - 1 - Z][
                                blob_size[0] - 1 - Y]
                        elif select[channel] == 17:
                            spiral_image_3channels[row_index][column_index][channel] = blob[blob_size[0] - 1 - Y][blob_size[0] - 1 - X][
                                blob_size[0] - 1 - Z]
                        elif select[channel] == 18:
                            spiral_image_3channels[row_index][column_index][channel] = blob[blob_size[0] - 1 - Z][blob_size[0] - 1 - Y][
                                blob_size[0] - 1 - X]
                        else:
                            print(select)
                            print('wrong')
                elif spiral_augmentation == '6Aug':
                    for channel in range(0, 3):
                        if select[channel] == 1:
                            spiral_image_3channels[row_index][column_index][channel] = blob[X][Y][Z]
                        elif select[channel] == 2:
                            spiral_image_3channels[row_index][column_index][channel] = blob[X][Z][Y]
                        elif select[channel] == 3:
                            spiral_image_3channels[row_index][column_index][channel] = blob[blob_size[0] - 1 - X][Y][Z]
                        elif select[channel] == 4:
                            spiral_image_3channels[row_index][column_index][channel] = blob[blob_size[0] - 1 - X][Z][Y]
                        elif select[channel] == 5:
                            spiral_image_3channels[row_index][column_index][channel] = blob[blob_size[0] - 1 - X][blob_size[0] - 1 - Y][
                                blob_size[0] - 1 - Z]
                        elif select[channel] == 6:
                            spiral_image_3channels[row_index][column_index][channel] = blob[blob_size[0] - 1 - X][blob_size[0] - 1 - Z][
                                blob_size[0] - 1 - Y]
                        else:
                            print(select)
                            print('wrong')
                else:
                     print('wrong')

                # # print(select)


    return spiral_image_3channels


def data_augmentation(blob):

    train_transforms = Compose(
        [AddChannel(),
         RandAffine(prob=0.5, translate_range=(8, 8, 8), padding_mode="border", as_tensor_output=False),
         RandFlip(prob=0.5, spatial_axis=0),
         RandFlip(prob=0.5, spatial_axis=1),
         RandFlip(prob=0.5, spatial_axis=2),
         RandRotate(range_x=90, range_y=90, range_z=90, prob=0.5),
         RandZoom(0.8, 1.2)
         ])
    data_trans = apply_transform(train_transforms, blob.astype(np.float))
    array = data_trans.squeeze(0)


    # array = blob
    #
    # ''' Random rotate '''
    # if random.choice([True, False]):
    #     rotate_angle = random.randint(0, 360)
    #     rotate_axis = random.sample([0, 1, 2], k=2)
    #     array = rotate(array, rotate_angle, axes=(rotate_axis[0], rotate_axis[1]), mode='mirror', reshape=False)

    # ''' Random roll '''
    # if random.choice([True, False]):
    #     shift_ = (random.randint(0, 63), random.randint(0, 63), random.randint(0, 63))
    #     array = np.roll(array, shift_, axis=(0, 1, 2))

    # """shift array by specified range along specified axis(x, y or z)"""
    # if random.choice([True, False]):
    #     shift_lst = [0] * array.ndim
    #     data_shape = array.shape
    #     shift_axis = random.randint(0, 2)
    #     shift_range = (np.random.rand() - 0.5) * 0.5
    #     shift_lst[shift_axis] = math.floor(
    #         shift_range * data_shape[shift_axis])
    #     array = shift(array, shift=shift_lst, mode='mirror', cval=0)

    return array


def load_from_preprocessed(file_name):
    epoch = random.randint(0, 5)
    if epoch == 5:
        cube = np.load(file_name)
    else:
        new_file_name = file_name.replace('normalized_cubic_npy_5subsets', 'normalized_cubic_npy_5subsets_reaug')
        new_file_name = new_file_name.replace('.npy', '_randomepoch_aug_{}.npy'.format(epoch))
        cube = np.load(new_file_name)
    return cube


class DataGenerator(Dataset):

    def __init__(self, list_IDs, method, shuffle=True, augmentation=False):
        'Initialization'
        # self.dim = dim
        self.list_IDs = list_IDs
        self.method = method  # 2D/ 2D_TL/ 2.5D/ 2.5D_TL/ 2.75D/ 2.75D_TL/ 2.75D_3channel/ 2.75D_3channel_TL/ 3D
        # self.n_channels = n_channels
        # self.n_classes = n_classes
        self.shuffle = shuffle
        # self.shuffle = False
        self.augmentation = augmentation
        # self.num_epochs = num_epochs
        self.indexes = np.arange(len(self.list_IDs))
        self.indexes_orig = np.arange(len(self.list_IDs))

        self.blob_size = 64
        self.number_of_planes = 9
        self.spiral_image_height = 32

        if self.shuffle:
            self.train_size_per_epoch = len(list_IDs) * 5 // num_epochs

        self.on_epoch_end()
        # self.transform = transform

    def __getitem__(self, index):

        img_ID = self.list_IDs[self.indexes[index]]
        # if self.augmentation:
        #     cube = load_from_preprocessed(img_ID)
        # else:
        #     cube = np.load(img_ID)

        cube = np.load(img_ID)
        # if self.augmentation:
        #     cube =data_augmentation(cube)

        img = self.__process_img(cube)
        img = torch.from_numpy(img.astype(np.float32))

        if 'real_' in img_ID:
            label = np.asarray(1, dtype='int64')
        elif 'fake_' in img_ID:
            label = np.asarray(0, dtype='int64')
        else:
            raise Exception('Filename does not contain expected keywords. Label was not assigned properly')

        return {'img': img, 'label': label}

    def __process_img(self, cube):
        method = self.method
        # # 2D/ 2D_TL/ 2.5D/ 2.5D_TL/ 2.75D/ 2.75D_TL/ 2.75D_3channel/ 2.75D_3channel_TL/ 3D

        if method == '2D':
            img = np.expand_dims(get_slices_from_blob(cube, self.blob_size)[:, :, 0], axis=0)
        elif method == '2D_TL':
            img = np.expand_dims(get_slices_from_blob(cube, self.blob_size)[:, :, 0], axis=0)
            img = np.concatenate((img, img, img), axis=0)
        elif method == '2.5D':
            img = np.expand_dims(get_slices_from_blob(cube, self.blob_size), axis=0)    # 1 * 64 * 64 * 9
            img = img.transpose(0,3,1,2)                                                # 1 * 9 * 64 * 64
            # img = np.reshape(get_slices_from_blob(cube, self.blob_size),
            # (1, self.blob_size, self.blob_size, self.number_of_planes))

        elif method == '2.5D_TL':
            img = np.expand_dims(get_slices_from_blob(cube, self.blob_size), axis=0)    # 1 * 64 * 64 * 9
            img = np.concatenate((img, img, img), axis=0)                               # 3 * 64 * 64 * 9
            # img = np.expand_dims(img, axis=0)                                           # 1 * 3 * 64 * 64 * 9
            img = img.transpose(0, 3, 1, 2)                                          # ->  1 * 9 * 3 * 64 * 64

        elif method == '2.75D':
            img = np.expand_dims(get_spiral_image(cube, self.spiral_image_height).squeeze(2), axis=0)  # 1 * 32 * 123
        elif method == '2.75D_TL':
            img = np.expand_dims(get_spiral_image(cube, self.spiral_image_height).squeeze(2), axis=0)
            img = np.concatenate((img, img, img), axis=0)
        elif method == '2.75D_3channel':
            img = np.expand_dims(get_spiral_image_3channels(cube, self.spiral_image_height), axis=0)  # 1 * 32 * 123 * 3
            img = img.transpose(0, 3, 1, 2)  # 1 * 3 * 32 * 123
        elif method == '2.75D_3channel_TL':
            img = np.expand_dims(get_spiral_image_3channels(cube, self.spiral_image_height), axis=0)  # 1 * 32 * 123 * 3
            img = np.concatenate((img, img, img), axis=0)                                   # 3 * 32 * 123 * 3
            # img = np.expand_dims(img, axis=0)                                               # 1 * 3 * 32 * 123 * 3
            # img = img.transpose(0, 4, 1, 2, 3)                                              # ->  1 * 3 * 3 * 32 * 123
            img = img.transpose(0, 3, 1, 2)
        elif method == '3D':
            img = np.expand_dims(cube, axis=0)
        else:
            raise Exception('Method {} is not supported'.format(method))

        return img

    def __len__(self):
        return len(self.indexes)
        # return self.train_size_per_epoch

    def on_epoch_end(self):
        if self.shuffle:
            'Updates indexes after each epoch'
            # self.indexes = np.arange(len(self.list_IDs))
            np.random.shuffle(self.indexes_orig)
            self.indexes = self.indexes_orig[:self.train_size_per_epoch]
        # print("Epoch sample size: ", len(self.indexes))

class MixedDataGenerator(Dataset):

    def __init__(self, list_IDs, method, shuffle=True, augmentation=False):
        'Initialization'
        # self.dim = dim
        self.list_IDs = list_IDs
        self.method = method  # 2D/ 2D_TL/ 2.5D/ 2.5D_TL/ 2.75D/ 2.75D_TL/ 2.75D_3channel/ 2.75D_3channel_TL/ 3D
        # self.n_channels = n_channels
        # self.n_classes = n_classes
        self.shuffle = shuffle
        # self.shuffle = False
        self.augmentation = augmentation
        # self.num_epochs = num_epochs
        self.indexes = np.arange(len(self.list_IDs))
        self.indexes_orig = np.arange(len(self.list_IDs))

        self.blob_size = 64
        self.number_of_planes = 9
        self.spiral_image_height = 32

        if self.shuffle:
            self.train_size_per_epoch = len(list_IDs) * 5 // num_epochs

        self.on_epoch_end()
        # self.transform = transform

    def __getitem__(self, index):

        img_ID = self.list_IDs[self.indexes[index]]
        if self.augmentation:
            cube = load_from_preprocessed(img_ID)
        else:
            cube = np.load(img_ID)
        # cube = np.load(img_ID)
        # if self.augmentation:
        #     cube =data_augmentation(cube)

        img, img_spiral = self.__process_img(cube)
        img = torch.from_numpy(img.astype(np.float32))
        img_spiral = torch.from_numpy(img_spiral.astype(np.float32))

        if 'real_' in img_ID:
            label = np.asarray(1, dtype='int64')
        elif 'fake_' in img_ID:
            label = np.asarray(0, dtype='int64')
        else:
            raise Exception('Filename does not contain expected keywords. Label was not assigned properly')

        return {'img': img,
                'img_spiral': img_spiral,
                'label': label}

    def __process_img(self, cube):
        method = self.method
        # # 2D/ 2D_TL/ 2.5D/ 2.5D_TL/ 2.75D/ 2.75D_TL/ 2.75D_3channel/ 2.75D_3channel_TL/ 3D

        if method == 'mixed_2D':
            img = np.expand_dims(get_slices_from_blob(cube, self.blob_size)[:, :, 0], axis=0)
            img_spiral = np.expand_dims(get_spiral_image(cube, self.spiral_image_height).squeeze(2), axis=0)  # 1 * 32 * 123
        elif method == 'mixed_2D_TL':
            img = np.expand_dims(get_slices_from_blob(cube, self.blob_size)[:, :, 0], axis=0)
            img = np.concatenate((img, img, img), axis=0)
            img_spiral = np.expand_dims(get_spiral_image(cube, self.spiral_image_height).squeeze(2), axis=0)
            img_spiral = np.concatenate((img_spiral, img_spiral, img_spiral), axis=0)

        elif method == 'mixed_2.5D':
            img = np.expand_dims(get_slices_from_blob(cube, self.blob_size), axis=0)    # 1 * 64 * 64 * 9
            img = img.transpose(0,3,1,2)                                                # 1 * 9 * 64 * 64
            img_spiral = np.expand_dims(get_spiral_image(cube, self.spiral_image_height).squeeze(2),
                                        axis=0)  # 1 * 32 * 123
            # img = np.reshape(get_slices_from_blob(cube, self.blob_size),
            # (1, self.blob_size, self.blob_size, self.number_of_planes))

        elif method == 'mixed_2.5D_TL':
            img = np.expand_dims(get_slices_from_blob(cube, self.blob_size), axis=0)    # 1 * 64 * 64 * 9
            img = np.concatenate((img, img, img), axis=0)                               # 3 * 64 * 64 * 9
            # img = np.expand_dims(img, axis=0)                                           # 1 * 3 * 64 * 64 * 9
            img = img.transpose(0, 3, 1, 2)                                          # ->  1 * 9 * 3 * 64 * 64
            img_spiral = np.expand_dims(get_spiral_image(cube, self.spiral_image_height).squeeze(2), axis=0)
            img_spiral = np.concatenate((img_spiral, img_spiral, img_spiral), axis=0)

        elif method == 'mixed_3channels_2D':
            img = np.expand_dims(get_slices_from_blob(cube, self.blob_size)[:, :, 0], axis=0)
            img_spiral = np.expand_dims(
                get_spiral_image_3channels(cube, self.spiral_image_height), axis=0)  # 1 * 32 * 123 * 3
            img_spiral = img_spiral.transpose(0, 3, 1, 2)  # 1 * 3 * 32 * 123

        elif method == 'mixed_3channels_2D_TL':
            img = np.expand_dims(get_slices_from_blob(cube, self.blob_size)[:, :, 0], axis=0)
            img = np.concatenate((img, img, img), axis=0)
            img_spiral = np.expand_dims(get_spiral_image_3channels(cube, self.spiral_image_height), axis=0)  #1*32*123*3
            img_spiral = np.concatenate((img_spiral, img_spiral, img_spiral), axis=0)  # 3 * 32 * 123 * 3
            img_spiral = img_spiral.transpose(0, 3, 1, 2)

        elif method == 'mixed_3channels_2.5D':
            img = np.expand_dims(get_slices_from_blob(cube, self.blob_size), axis=0)    # 1 * 64 * 64 * 9
            # img = np.expand_dims(img, axis=0)                                           # 1 * 3 * 64 * 64 * 9
            img = img.transpose(0, 3, 1, 2)                                          # ->  1 * 9 * 3 * 64 * 64
            img_spiral = np.expand_dims(
                get_spiral_image_3channels(cube, self.spiral_image_height), axis=0)  # 1 * 32 * 123 * 3
            img_spiral = img_spiral.transpose(0, 3, 1, 2)

        elif method == 'mixed_3channels_2.5D_TL':
            img = np.expand_dims(get_slices_from_blob(cube, self.blob_size), axis=0)    # 1 * 64 * 64 * 9
            img = np.concatenate((img, img, img), axis=0)                               # 3 * 64 * 64 * 9
            # img = np.expand_dims(img, axis=0)                                         # 1 * 3 * 64 * 64 * 9
            img = img.transpose(0, 3, 1, 2)                                             # ->  1 * 9 * 3 * 64 * 64
            img_spiral = np.expand_dims(get_spiral_image_3channels(cube, self.spiral_image_height),axis=0)  # 1*32*123*3
            img_spiral = np.concatenate((img_spiral, img_spiral, img_spiral), axis=0)  # 3*32*123*3
            img_spiral = img_spiral.transpose(0, 3, 1, 2)

        else:
            raise Exception('Method {} is not supported'.format(method))

        return img, img_spiral

    def __len__(self):
        return len(self.indexes)
        # return self.train_size_per_epoch

    def on_epoch_end(self):
        if self.shuffle:
            'Updates indexes after each epoch'
            # self.indexes = np.arange(len(self.list_IDs))
            np.random.shuffle(self.indexes_orig)
            self.indexes = self.indexes_orig[:self.train_size_per_epoch]
        # print("Epoch sample size: ", len(self.indexes))


def dataloader(list_IDs, method, shuffle, augmentation, batch_size, num_workers):

    if 'mixed' in method:
        dataset = MixedDataGenerator(list_IDs, method, shuffle=shuffle, augmentation=augmentation)
    else:
        dataset = DataGenerator(list_IDs, method, shuffle=shuffle, augmentation=augmentation)

    data_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle,
                             num_workers=num_workers, pin_memory=True, drop_last=True)

    return data_loader





