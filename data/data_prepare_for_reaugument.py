# -*- coding:utf-8 -*-

import argparse
import numpy as np
import sys
from pathlib import Path
from monai.transforms import AddChannel, Compose,apply_transform, \
    RandAffine, RandZoom, RandRotate, RandFlip, \
    RandGaussianNoise, RandGaussianSharpen, RandGaussianSmooth, RandHistogramShift
import os

def arg_parse():
    parser = argparse.ArgumentParser(description='')

    parser.add_argument('--base_dir', default='/processing/to/data/RESULT/', type=str,  help='')

    parser.add_argument('--only-one-subset', action='store_true', default=True, help='')

    parser.add_argument('--subset', default=1, type=int, metavar='N',
                        help='0 - 4')

    parser.add_argument('--subsets', default=5, type=int, metavar='N',
                        help='5 or 10')

    parser.add_argument('--epochs', default=5, type=int, metavar='N',
                        help='')

    parser.add_argument('--epoch', default=0, type=int, metavar='N',
                        help='0-(epochs-1)')

    parser.add_argument('--only-one-epoch', action='store_true',
                        default=True,
                        help='')

    # For balance
    # ---------------------------------------------------------------
    parser.add_argument('--data-balance', action='store_true',
                        default=False,
                        help='')

    parser.add_argument('--data-aug', action='store_true',
                        default=True,
                        help='')

    # For data Aug
    # ---------------------------------------------------------------
    parser.add_argument('--RandZoom', action='store_true',
                        default=True,
                        help='')
    parser.add_argument('--RandFlipX', action='store_true',
                        default=True,
                        help='')
    parser.add_argument('--RandFlipY', action='store_true',
                        default=True,
                        help='')
    parser.add_argument('--RandFlipZ', action='store_true',
                        default=True,
                        help='')
    parser.add_argument('--RandRotate', action='store_true',
                        default=True,
                        help='')
    parser.add_argument('--RandAffine', action='store_true',
                        default=True,
                        help='')
    parser.add_argument('--RandGaussianNoise', action='store_true',
                        default=True,
                        help='')
    parser.add_argument('--RandGaussianSharpen', action='store_true',
                        default=True,
                        help='')
    parser.add_argument('--RandGaussianSmooth', action='store_true',
                        default=True,
                        help='')
    parser.add_argument('--RandHistogramShift', action='store_true',
                        default=True,
                        help='')
    # ---------------------------------------------

    args = parser.parse_args()
    return args


def generate_data(file, label_index, args):
    '''
     @param file : a npy file which store all information of one cubic
     @param label_index: file name suffix of the augmented file
    '''
    array = np.load(file)

    transforms_list = [AddChannel()]

    if args.RandAffine:
        transforms_list.append(
            RandAffine(prob=0.5, translate_range=(8, 8, 8), padding_mode="border", as_tensor_output=False))

    if args.RandFlipX:
        transforms_list.append(RandFlip(prob=0.5, spatial_axis=0))

    if args.RandFlipY:
        transforms_list.append(RandFlip(prob=0.5, spatial_axis=1))

    if args.RandFlipZ:
        transforms_list.append(RandFlip(prob=0.5, spatial_axis=2))

    if args.RandRotate:
        transforms_list.append(RandRotate(range_x=90, range_y=90, range_z=90, prob=0.5))

    if args.RandZoom:
        transforms_list.append(RandZoom(0.75, 1.25))

    if args.RandGaussianNoise:
        transforms_list.append(RandGaussianNoise(prob=0.5))

    if args.RandGaussianSharpen:
        transforms_list.append(RandGaussianNoise(prob=0.5))

    if args.RandGaussianSmooth:
        transforms_list.append(RandGaussianNoise(prob=0.5))

    if args.RandHistogramShift:
        transforms_list.append(RandHistogramShift(prob=0.5))

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

    np.save(file.replace(".npy", "_random" + str(label_index) + ".npy"), array)




# def data_balance(normalization_output_path, new_normalization_output_path, args):
#     '''
#     Augmentation
#     '''
#     files = [os.path.join(normalization_output_path, x) for x in os.listdir(normalization_output_path) if
#              "_size64x64.npy" in x]
#
#     for file in files:
#         array = np.load(file)
#         file = file.replace(normalization_output_path, new_normalization_output_path)
#         np.save(file, array)
#         print(file, '---resaved')
#
#     files = [os.path.join(new_normalization_output_path, x) for x in os.listdir(new_normalization_output_path)]
#
#     print("number of candidate nodules under normalization path %s:  " % normalization_output_path, len(files))
#     real_files = [m for m in files if "real" in m]
#     print("Before augmentation: number of real nodules under normalization path %s:  " % normalization_output_path,
#           len(real_files))
#     fake_files = [m for m in files if "fake" in m]
#     print(len(fake_files))
#     print(len(real_files))
#
#     nr_aug_loops = len(fake_files) // len(real_files)
#     remainder = len(fake_files) % len(real_files)
#
#     for aug in range(1, nr_aug_loops):
#         for file in real_files:
#             print("Augment index: ", str(aug), "for file: ", file)
#             generate_data(file, 'balance_aug_{}'.format(aug), args)
#
#     for file_index, file in enumerate(real_files):
#         if file_index == remainder:
#             break
#         generate_data(file, 'balance_aug_{}'.format(nr_aug_loops), args)

def data_balance(normalization_output_path, new_normalization_output_path, args):
    '''
    Augmentation
    '''
    files = [os.path.join(normalization_output_path, x) for x in os.listdir(normalization_output_path)]

    for file in files:
        array = np.load(file)
        file = file.replace(normalization_output_path, new_normalization_output_path)
        np.save(file, array)
        print(file, '---resaved')


def data_aug(new_normalization_output_path, args):
    files = [os.path.join(new_normalization_output_path, x) for x in os.listdir(new_normalization_output_path)]

    if not args.only_one_epoch:
        epochs = range(args.epochs)
    else:
        epochs = range(args.epoch, args.epoch + 1)

    for epoch in epochs:
        for file in files:
            print("Augment index: ", str(epoch), "for file: ", file)
            generate_data(file, 'epoch_aug_{}'.format(epoch), args)

    print("Data augmentation for subset ", str(subset), " is completed successfully")


if __name__ == '__main__':

    args = arg_parse()
    base_dir = args.base_dir
    # normalized_volume_path = args.normalized_volume_path
    normalized_volume_path = base_dir + 'normalized_cubic_npy_5subsets/'
    #args.subset = ['0']  #  xin_sub

    if not args.only_one_subset:
        subsets = range(args.subsets)
    else:
        subsets = range(args.subset, args.subset+1)

        for subset in subsets:
            # dcim_path = original_dicom_data_dirpath + 'subset' + str(subset) + "/"
            # plot_output_path = plot_output_path_base + 'subset' + str(subset)
            normalization_output_path = normalized_volume_path + 'subset' + str(subset)
            new_normalized_volume_path = base_dir + 'normalized_cubic_npy_5subsets_reaug/'
            Path(new_normalized_volume_path).mkdir(parents=True, exist_ok=True)
            new_normalization_output_path = new_normalized_volume_path + 'subset' + str(subset)
            # error_candidates_path = error_candidates_path_base + 'subset' + str(subset)
            print("extracting image into %s" % normalization_output_path)
            # if not os.path.exists(plot_output_path):
            #     os.mkdir(plot_output_path)
            # if not os.path.exists(normalization_output_path):
            #     os.mkdir(normalization_output_path)
            if not os.path.exists(new_normalization_output_path):
                os.mkdir(new_normalization_output_path)
            # extract_candidate_cubic_from_mhd(dcim_path, candidate_file, plot_output_path, normalization_output_path,
            #                                  error_candidates_path)

            # print("candidate nodule extraction from subset ", str(subset), " is completed successfully")
            if args.data_balance:
                data_balance(normalization_output_path, new_normalization_output_path, args)

            if args.data_aug:
                data_aug(new_normalization_output_path, args)

            print('Subset {}: Finished'.format(subset))

    print('Finished')



