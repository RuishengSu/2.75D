# -*- coding:utf-8 -*-

import argparse
import SimpleITK as sitk
import numpy as np
from glob import glob
import pandas as pd
import matplotlib.pyplot as plt
import traceback
import random
from scipy.ndimage.interpolation import shift, zoom
from scipy.ndimage import rotate
from skimage.transform import resize
import math
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
import cv2 as cv
from pathlib import Path

MIN_BOUND = -1000.0
MAX_BOUND = 400.0


def get_filename(file_list, case):
    for f in file_list:
        if case in f:
            return f


def get_cube_index_range(center, spacing, image_size):
    '''
    @param: center: nodule center in voxel space (still x,y,z ordering)
    @param: spacing:  spacing of voxels in world coor. (mm) (x, y,z ordering)
    @param: image_size: size of the entire image which the cube is extracted from
    '''
    cube_height = int(50 // spacing[0] + 1)  # size of the 50x50x50 mm cube
    cube_width = int(50 // spacing[1] + 1)
    cube_thickness = int(50 // spacing[2] + 1)

    x_end = min(image_size[0] - 1, int(center[0] + cube_height // 2))
    x_start = x_end - cube_height + 1
    # if x_start < 0:
    #     x_start = 0
    #     x_end = cube_height - 1

    y_end = min(image_size[1] - 1, int(center[1] + cube_width // 2))
    y_start = y_end - cube_width + 1
    # if y_start < 0:
    #     y_start = 0
    #     y_end = cube_width - 1

    z_end = min(image_size[2] - 1, int(center[2] + cube_thickness // 2))
    z_start = z_end - cube_thickness + 1
    # if z_start < 0:
    #     z_start = 0
    #     z_end = cube_thickness - 1

    return x_start, x_end, y_start, y_end, z_start, z_end


def load_itk_image(filename):
    with open(filename) as f:
        contents = f.readlines()
        line = [k for k in contents if k.startswith('TransformMatrix')][0]
        transformM = np.array(line.split(' = ')[1].split(' ')).astype('float')
        transformM = np.round(transformM)
        if np.any(transformM != np.array([1, 0, 0, 0, 1, 0, 0, 0, 1])):
            isflip = True
        else:
            isflip = False

    itkimage = sitk.ReadImage(filename)
    numpyImage = sitk.GetArrayFromImage(itkimage)

    numpyOrigin = np.array(list(itkimage.GetOrigin()))
    numpySpacing = np.array(list(itkimage.GetSpacing()))

    return numpyImage, numpyOrigin, numpySpacing, isflip


def extract_candidate_cubic_from_mhd(dcim_path, candidate_file, plot_output_path, normalization_output_path,
                                     error_candidates_path):
    '''
      @param: dcim_path :                 the path contains all mhd file
      @param: candidate_file:             the candidate csv file,contains every **fake** nodules' coordinate
      @param: plot_output_path:           the save path of extracted cubic of size 64x64x64 npy file(plot )
      @param: normalization_output_path:  the save path of extracted cubic of size 64x64x64 npy file(after normalization)
      @param: error_candidates_path:      the save path of extracted cubic of size 64x64x64 npy file that contain NaN values
    '''
    file_list = glob(dcim_path + "*.mhd")
    # The locations of the nodes
    df_node = pd.read_csv(candidate_file)
    df_node["file"] = df_node["seriesuid"].map(lambda file_name: get_filename(file_list, file_name))
    df_node = df_node.dropna()

    for img_file in file_list:
        mini_df = df_node[df_node["file"] == img_file]  # get all nodules associate with file
        print("shape of mini_df: ", mini_df.shape)
        print("mini_df describe below:")
        print(mini_df.describe())
        # file_name = str(img_file).split("/")[-1]
        file_name = os.path.basename(img_file)
        if mini_df.shape[0] > 0:  # some files may not have a nodule--skipping those
            # load the data once
            img_array, origin, spacing, isflip = load_itk_image(img_file)
            # if isflip:
            #     img_array = img_array[:, ::-1, ::-1]
            #     print('flip!')
            num_z, height, width = img_array.shape  # heightXwidth constitute the transverse plane
            cube_height = int(50 // spacing[0] + 1)
            #            cube_width = int(50//spacing[1] + 1)  # size of the 50x50x50 mm cube
            #            cube_thickness = int(50//spacing[2] + 1)
            print("extracting ", str(mini_df.shape[0]), " nodes from file: ", img_file)

            # go through all nodes
            print("begin to process candidate nodules...")
            img_array = normalize(img_array).astype(np.float32)
            img_array = img_array.transpose(2, 1, 0)  # transform from z, y, x to x,y,z
            print(img_array.shape)
            pad_x = int(np.ceil(32/spacing[0]))
            pad_y = int(np.ceil(32/spacing[1]))
            pad_z = int(np.ceil(32/spacing[2]))
            img_array_padded = np.pad(img_array, [(pad_x, ), (pad_y, ), (pad_z, )], mode='constant')
            for node_idx, cur_row in mini_df.iterrows():
                node_x = cur_row["coordX"]
                node_y = cur_row["coordY"]
                node_z = cur_row["coordZ"]
                node_class = cur_row["class"]
                nodule_pos_str = str(node_x) + "_" + str(node_y) + "_" + str(node_z)
                w_center = np.array([node_x, node_y, node_z])  # nodule center
                v_center = worldToVoxelCoord(w_center, origin, spacing)
                # if isflip:
                #     v_center[0] = height - v_center[0]
                #     v_center[1] = width - v_center[1]
                # imgs = np.ndarray([64, 64, 64], dtype=np.float32)
                try:
                    x_start, x_end, y_start, y_end, z_start, z_end = get_cube_index_range(v_center, spacing,
                                                                                          img_array.shape)
                    x_start, x_end, y_start, y_end, z_start, z_end = x_start+pad_x, x_end+pad_x, y_start+pad_y, y_end+pad_y, z_start+pad_z, z_end+pad_z
                    if x_start < 0 or y_start < 0 or z_start < 0 or x_end >= img_array_padded.shape[0] or y_end >= \
                            img_array_padded.shape[1] or z_end >= img_array_padded.shape[2]:
                        print("Out of Range!! Padded image shape: ", img_array_padded.shape,
                              " Cube range [x_start, x_end, y_start, y_end, z_start, z_end]: ", x_start, x_end, y_start,
                              y_end, z_start, z_end)
                    if x_start < pad_x or y_start < pad_y or x_end >= img_array_padded.shape[0] - pad_x or y_end >= \
                            img_array_padded.shape[1]-pad_y:
                        v_center_padded = v_center + [pad_x, pad_y, pad_z]
                        vis = (img_array_padded[:, :, int(v_center_padded[2])]).copy()
                        vis = cv.rectangle(vis*255, (y_start, x_start), (y_end, x_end), 255, 3)
                        cv.imwrite(os.path.join(base_dir, 'visualization/edge', "{}.bmp".format(file_name)), vis)
                    imgs = resize(img_array_padded[x_start:x_end, y_start:y_end, z_start:z_end], (64, 64, 64),
                                           mode='constant', cval=0)

                    if node_class == 1:
                        v_center_padded = v_center + [pad_x, pad_y, pad_z]
                        vis = (img_array_padded[:, :, int(v_center_padded[2])]).copy()
                        vis = cv.rectangle(vis*255, (y_start, x_start), (y_end, x_end), 255, 3)
                        cv.imwrite(os.path.join(base_dir, 'visualization/real', "%s_pos%s_%d_real_size64.bmp" % (
                            str(file_name), nodule_pos_str, node_idx)), vis)
                        cv.imwrite(os.path.join(base_dir, 'visualization/volume', "%s_pos%s_%d_real_size64.bmp" % (
                            str(file_name), nodule_pos_str, node_idx)), imgs[:, :, 32]*255)

                    if np.isnan(imgs).any():
                        if node_class == 1:
                            path_to_save = os.path.join(error_candidates_path, "%s_pos%s_%d_real_size64x64.npy" % (
                            str(file_name), nodule_pos_str, node_idx))
                            np.save(path_to_save, imgs)
                            raise Exception("Error!: NaN value in %s_pos%s_%d_real_size64x64.npy" % (
                            str(file_name), nodule_pos_str, node_idx))
                        if node_class == 0:
                            np.save(os.path.join(error_candidates_path, "%s_pos%s_%d_fake_size64x64.npy" % (
                            str(file_name), nodule_pos_str, node_idx)), imgs)
                            raise Exception("Error!: NaN value in %s_pos%s_%d_fake_size64x64.npy" % (
                            str(file_name), nodule_pos_str, node_idx))
                    else:
                        if node_class == 1:
                            np.save(os.path.join(normalization_output_path, "%s_pos%s_%d_real_size64x64.npy" % (
                            str(file_name), nodule_pos_str, node_idx)), imgs)
                        elif node_class == 0:
                            np.save(os.path.join(normalization_output_path, "%s_pos%s_%d_fake_size64x64.npy" % (
                            str(file_name), nodule_pos_str, node_idx)), imgs)

                except Exception as e:
                    print(" process images %s error..." % str(file_name))
                    print(Exception, ":", e)
                    traceback.print_exc()


def plot_cubic(npy_file, save_path):
    '''
       plot the cubic slice by slice

    :param npy_file:
    :return:
    '''
    ncol = 8
    cubic_array = np.load(npy_file)
    f, plots = plt.subplots(int(cubic_array.shape[2] / ncol), ncol, figsize=(50, 50))
    for i in range(0, cubic_array.shape[2]):
        plots[int(i / ncol), int((i % ncol))].axis('off')
        plots[int(i / ncol), int((i % ncol))].imshow(cubic_array[:, :, i], cmap='gray')
    # plt.show()
    plt.savefig(save_path)
    plt.close(f)


def plot_3d_cubic(image):
    '''
        plot the 3D cubic
    :param image:   image saved as npy file path
    :return:
    '''
    from skimage import measure
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    image = np.load(image)
    verts, faces, norm, val = measure.marching_cubes_lewiner(image, 0)
    fig = plt.figure(figsize=(40, 40))
    ax = fig.add_subplot(111, projection='3d')
    # Fancy indexing: `verts[faces]` to generate a collection of triangles
    mesh = Poly3DCollection(verts[faces], alpha=0.1)
    face_color = [0.5, 0.5, 1]
    mesh.set_facecolor(face_color)
    ax.add_collection3d(mesh)
    ax.set_xlim(0, image.shape[0])
    ax.set_ylim(0, image.shape[1])
    ax.set_zlim(0, image.shape[2])
    plt.show()


def normalize(image):
    maxHU = 400.
    minHU = -1000.
    image[image > maxHU] = maxHU
    image[image < minHU] = minHU
    image = (image - minHU) / (maxHU - minHU)
    return image


def worldToVoxelCoord(worldCoord, origin, spacing):
    stretchedVoxelCoord = np.absolute(worldCoord - origin)
    voxelCoord = stretchedVoxelCoord / spacing
    return voxelCoord


def search(path, word):
    '''
    find filename match keyword from path
    :param path:  path search from
    :param word:  keyword should be matched
    :return:
    '''
    filelist = []
    for filename in os.listdir(path):
        fp = os.path.join(path, filename)
        if os.path.isfile(fp) and word in filename:
            filelist.append(fp)
        elif os.path.isdir(fp):
            search(fp, word)
    return filelist


def get_all_filename(path, size):
    list_real = search(path, 'real_size' + str(size) + "x" + str(size))
    list_fake = search(path, 'fake_size' + str(size) + "x" + str(size))
    return list_real + list_fake


def zoom_data(data):
    """zoom array by specified range along all. After zoomed, the voxel size is the same as
    before"""
    data_shape = data.shape
    data_dim = data.ndim
    # functions to calculate target range of arrays(outside of the target range is not used to zoom)
    # - d/2 <= zoom_range * (x - d/2) <= d/2
    zoom_range = np.random.uniform(1, 1.25)
    zoom_axis = None
    f1 = lambda d: math.ceil((d / 2) * (1 + 1 / zoom_range))
    f2 = lambda d: math.floor((d / 2) * (1 - 1 / zoom_range))

    # expand
    z_win1 = list(map(f1, data_shape[:]))
    z_win2 = list(map(f2, data_shape[:]))
    if zoom_axis is None:
        # same for all axis
        target_data = data[z_win2[0]:z_win1[0], z_win2[1]:z_win1[1], z_win2[2]:z_win1[2]]
    else:
        # only one axis
        if zoom_axis == 0:
            target_data = data[z_win2[0]:z_win1[0], :, :]
        elif zoom_axis == 1:
            target_data = data[:, z_win2[1]:z_win1[1], :]
        elif zoom_axis == 2:
            target_data = data[:, :, z_win2[2]:z_win1[2]]

    if zoom_axis is None:
        zoom_lst = [zoom_range] * data_dim
    else:
        zoom_lst = [1] * data_dim
        zoom_lst[zoom_axis] = zoom_range

    zoomed = zoom(target_data, zoom=zoom_lst, cval=0)
    if zoomed.shape[0] < 64 or zoomed.shape[1] < 64 or zoomed.shape[2] < 64:
        print("This is not intended behaviour!")
        zoomed = target_data
    if zoomed.shape[0] > 64:
        zoomed = zoomed[0:64, :, :]
    if zoomed.shape[1] > 64:
        zoomed = zoomed[:, 0:64, :]
    if zoomed.shape[2] > 64:
        zoomed = zoomed[:, :, 0:64]

    return zoomed


def generate_data(file, label_index):
    '''
     @param file : a npy file which store all information of one cubic
     @param label_index: file name suffix of the augmented file
    '''
    array = np.load(file)

    # false_true_index = random.randint(1, 4)

    ''' Random rotate '''
    # if random.choice([True, False]) or false_true_index == 1:
    #     rotate_angle = random.randint(0, 360)
    #     rotate_axis = random.sample([0, 1, 2], k=2)
    #     array = rotate(array, rotate_angle, axes=(rotate_axis[0], rotate_axis[1]), mode='mirror', reshape=False)

    '''Random flip'''
    if random.choice([True, False]):
        flip_axis = random.randint(0, 2)
        array = np.flip(array, flip_axis)
    '''Random zoom'''
    if random.choice([True, False]):
        array = zoom_data(array)

    # """shift array by specified range along specified axis(x, y or z)"""
    # if random.choice([True, False]) or false_true_index == 4:
    #     shift_lst = [0] * array.ndim
    #     data_shape = array.shape
    #     shift_axis = random.randint(0, 2)
    #     shift_range = (np.random.rand() - 0.5) * 0.5
    #     shift_lst[shift_axis] = math.floor(
    #         shift_range * data_shape[shift_axis])
    #     array = shift(array, shift=shift_lst, mode='mirror', cval=0)

    cv.imwrite(os.path.join(base_dir, 'visualization/volume', "{}_real_aug.bmp".format(
        str(Path(file).stem))), array[:, :, 32]*255)
    np.save(file.replace(".npy", "_random" + str(label_index) + ".npy"), array)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Extract 3D volumes of 64x64x64 from Luna dataset')
    parser.add_argument('--subset', type=int, nargs='+',
                        help='choose a subset folder from which the 3D volumes will be extracted')


    args = parser.parse_args()
    #args.subset = ['0']  #  xin_sub

    if args.subset is not None:
        subsets = args.subset
    else:
        subsets = range(0, 10)

    for subset in subsets:
        dcim_path = original_dicom_data_dirpath + 'subset' + str(subset) + "/"
        plot_output_path = plot_output_path_base + 'subset' + str(subset)
        normalization_output_path = normalized_volume_path + 'subset' + str(subset)
        error_candidates_path = error_candidates_path_base + 'subset' + str(subset)
        print("extracting image into %s" % normalization_output_path)
        if not os.path.exists(plot_output_path):
            os.mkdir(plot_output_path)
        if not os.path.exists(normalization_output_path):
            os.mkdir(normalization_output_path)
        if not os.path.exists(error_candidates_path):
            os.mkdir(error_candidates_path)
        extract_candidate_cubic_from_mhd(dcim_path, candidate_file, plot_output_path, normalization_output_path,
                                         error_candidates_path)

        print("candidate nodule extraction from subset ", str(subset), " is completed successfully")

        '''
        Augmentation
        '''
        files = [os.path.join(normalization_output_path, x) for x in os.listdir(normalization_output_path)]
        print("number of candidate nodules under normalization path %s:  " % normalization_output_path, len(files))
        real_files = [m for m in files if "real" in m]
        print("Before augmentation: number of real nodules under normalization path %s:  " % normalization_output_path,
              len(real_files))
        fake_files = [m for m in files if "fake" in m]
        print(len(fake_files))
        print(len(real_files))
        nr_aug_loops = len(fake_files) // len(real_files)
        remainder = len(fake_files) % len(real_files)

        for aug in range(1, nr_aug_loops):
            for file in real_files:
                print("Augment index: ", str(aug), "for file: ", file)
                generate_data(file, aug)

        for file_index, file in enumerate(real_files):
            if file_index == remainder:
                break
            generate_data(file, nr_aug_loops)

        print("Data augmentation for subset ", str(subset), " is completed successfully")
