import argparse
import numpy as np
import sys
from pathlib import Path
import math
import random
import matplotlib.pyplot as plt
from monai.transforms import AddChannel, Compose,apply_transform, \
    RandAffine, RandZoom, RandRotate, RandFlip, \
    RandGaussianNoise, RandGaussianSharpen, RandGaussianSmooth, RandHistogramShift

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

def generate_data(file, transform):
    '''
     @param file : a npy file which store all information of one cubic
     @param label_index: file name suffix of the augmented file
    '''
    array = np.load(file)

    transforms_list = [AddChannel()]

    prob = 1

    if transform == 'RandAffine':
        transforms_list.append(
            RandAffine(prob=prob, translate_range=(8, 8, 8), padding_mode="border", as_tensor_output=False))

    if transform == 'RandFlipX':
        transforms_list.append(RandFlip(prob=prob, spatial_axis=0))

    if transform == 'RandFlipY':
        transforms_list.append(RandFlip(prob=prob, spatial_axis=1))

    if transform == 'RandFlipZ':
        transforms_list.append(RandFlip(prob=prob, spatial_axis=2))

    if transform == 'RandRotate':
        transforms_list.append(RandRotate(range_x=90, range_y=90, range_z=90, prob=prob))

    if transform == 'RandZoom':
        transforms_list.append(RandZoom(0.75, 1.25))

    if transform == 'RandGaussianNoise':
        transforms_list.append(RandGaussianNoise(prob=prob))

    if transform == 'RandGaussianSharpen':
        transforms_list.append(RandGaussianSharpen(prob=prob))

    if transform == 'RandGaussianSmooth':
        transforms_list.append(RandGaussianSmooth(prob=prob))

    if transform == 'RandHistogramShift':
        transforms_list.append(RandHistogramShift(prob=prob))

    # train_transforms = Compose(
    #     [AddChannel(),
    #      RandAffine(prob=0.5, translate_range=(8, 8, 8), padding_mode="border", as_tensor_output=False),
    #      RandFlip(prob=0.5, spatial_axis=0),
    #      RandFlip(prob=0.5, spatial_axis=1),
    #      RandFlip(prob=0.5, spatial_axis=2),
    #      RandRotate(range_x=90, range_y=90, range_z=90, prob=0.5),
    #      RandZoom(0.8, 1.2)
    #      ])

    train_transforms = Compose(transforms_list)
    data_trans = apply_transform(train_transforms, array.astype(np.float))
    array = data_trans.squeeze(0)

    # np.save(file.replace(".npy", "_random" + str(label_index) + ".npy"), array)
    return array


file = '/processing/x.wang/Duke-data/RESULT/normalized_cubic_npy/subset0/' \
       'Breast_MRI_001_pos291.5_363.5_129.5_real_size64x64.npy'

transforms_list = [
    'RandAffine', 'RandFlipX', 'RandFlipY', 'RandFlipZ', 'RandRotate', 'RandZoom', 'RandGaussianNoise',
    'RandGaussianSharpen', 'RandGaussianSmooth', 'RandHistogramShift'
]
fig = plt.figure(figsize=(15, 10), dpi=300)
plt.axis('off')
for i in range(len(transforms_list)):
    array = generate_data(file, transforms_list[i])

    img_275d = get_spiral_image(array, 32)[:, :, 0]
    img_2d = get_slices_from_blob(array, 64)[:, :, 0]


    ax1 = fig.add_subplot(len(transforms_list)//2, 2 * 2, (i * 2 + 1))
    ax1.imshow(img_2d, cmap='gray')
    #plt.title('%s' % images_test_name)
    plt.title('2D_{}'.format(transforms_list[i]), fontsize=8)
    plt.axis('off')

    ax1 = fig.add_subplot(len(transforms_list)//2, 2 * 2, (i * 2 + 2))
    ax1.imshow(img_275d, cmap='gray')
    # plt.title('%s' % images_test_name)
    plt.title('2.75D_{}'.format(transforms_list[i]), fontsize=8)
    plt.axis('off')

fig.tight_layout()
plt.show()
# plt.close()
print('111')



