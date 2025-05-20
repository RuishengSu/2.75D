# -*- coding: utf-8 -*-
import os
import argparse
import pickle
import re
import numpy as np
import matplotlib.pyplot as plt
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
from training.training import training_loop

# os.environ["CUDA_VISIBLE_DEVICES"] = "7"
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def plot_auc(fpr, tpr, auc_score, label):
    plt.figure()
    plt.plot([0, 1], [0, 1], 'k--')
    plt.plot(fpr, tpr, label='Keras (area = {:.3f})'.format(auc_score))
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('ROC curve -- ' + label)
    plt.legend(loc='best')
    # plt.show()
    if not (reuse_results and os.path.isfile(args.result_path_base + label + '.png')):
        plt.savefig(args.result_path_base + label + '.png')


def prepare_input_filenames(testset='random', num_subsets=10):
    def downsample_filenames(filenames, downsample_ratio=training_data_downsample_ratio):
        if downsample_ratio == 1:
            return filenames
        negative_samples = [f for f in filenames if 'fake_' in f]
        positive_samples = [f for f in filenames if 'real_' in f]
        positive_original_samples = [f for f in positive_samples if '_random' not in f]
        downsampled_positive_original_samples = np.random.choice(positive_original_samples,
                                                                 len(positive_original_samples) // downsample_ratio,
                                                                 replace=False)

        downsampled_positive_samples = [f for f in positive_samples if
                                        re.sub('_random[0-9]*.npy', '.npy', f) in downsampled_positive_original_samples]
        downsampled_negative_samples = np.random.choice(negative_samples, len(downsampled_positive_samples),
                                                        replace=False)
        return downsampled_positive_samples + downsampled_negative_samples.tolist()

    filename_list_path = normalized_filelist_path

    training_filenames = np.array([])
    validation_filenames = np.array([])
    test_filenames = np.array([])

    if num_subsets == 10:
        subset_border1, subset_border2 = 7, 9
    elif num_subsets == 5:
        subset_border1, subset_border2 = 3, 4
    else:
        raise ValueError("Invalid number of subsets")
    if testset == 'random':
        subset_range = np.arange(num_subsets)
        np.random.shuffle(subset_range)
        splitted_indices = np.split(subset_range, [subset_border1, subset_border2])
    else:
        subset_range = np.arange(num_subsets)
        subset_range = np.delete(subset_range, int(testset))
        np.random.shuffle(subset_range)
        splitted_indices = np.split(subset_range, [subset_border1])
        splitted_indices.append(np.array([int(testset)]))

    ''' training filenames '''
    for subset_index in splitted_indices[0]:
        filename_npy_path = os.path.join(filename_list_path, 'subset' + str(subset_index) + '_filenames.npy')
        subset_filenames = np.load(filename_npy_path)
        training_filenames = np.append(training_filenames, subset_filenames)
    sampled_training_filenames = downsample_filenames(training_filenames)
    print("Testset {}: total training size: {}; sampled training size: {}".format(testset, len(training_filenames),
                                                                                  len(sampled_training_filenames)))
    '''validation filenames'''
    for subset_index in splitted_indices[1]:
        filename_npy_path = os.path.join(filename_list_path, 'subset' + str(subset_index) + '_filenames.npy')
        subset_filenames = np.load(filename_npy_path)
        validation_filenames = np.append(validation_filenames, subset_filenames)
        # subset_non_augmented_filenames = [file_path for file_path in np.load(filename_npy_path) if
        #                                   "_size64x64.npy" in file_path]
        # validation_filenames = np.append(validation_filenames, subset_non_augmented_filenames)
    sampled_validation_filenames = downsample_filenames(validation_filenames)
    print(
        "Testset {}: total validation size: {}; sampled validation size: {}".format(testset, len(validation_filenames),
                                                                                    len(sampled_validation_filenames)))

    '''test filenames'''
    for subset_index in splitted_indices[2]:
        filename_npy_path = os.path.join(filename_list_path, 'subset' + str(subset_index) + '_filenames.npy')
        subset_non_augmented_filenames = [file_path for file_path in np.load(filename_npy_path) if
                                          "_size64x64.npy" in file_path]
        test_filenames = np.append(test_filenames, subset_non_augmented_filenames)

    print("Testset {}: total test size: {}".format(testset, len(test_filenames)))
    #sys.stdout.flush()
    return sampled_training_filenames, sampled_validation_filenames, test_filenames


def arg_parse():
    parser = argparse.ArgumentParser(description='Run nodule classification on various data split')
    parser.add_argument('--testset', type=int, nargs='+',
                        help='choose test set(s) ')
    parser.add_argument('--result_path_base', type=str, default='./Result/',
                        help='Path to save results')

    parser.add_argument('--shuffle',
                        default=True, action='store_true',
                        help='')

    parser.add_argument('--batch_size', type=int,
                        default=64,
                        help='choose test set(s)')


    parser.add_argument('--method', default='2.5D', type=str,
                        help='2.75D, 2.75D_TL, 2.75D_3channel, 2.75D_3channel_TL')
    args = parser.parse_args()

    os.makedirs(args.result_path_base, exist_ok=True)
    args.result_path_base_new = args.result_path_base[0:-1] + '_last/'
    os.makedirs(args.result_path_base_new, exist_ok=True)

    args.sampled_filelist_path = args.result_path_base + 'filenames'
    os.makedirs(args.sampled_filelist_path, exist_ok=True)


    print(args)
    return args



if __name__ == '__main__':
    args = arg_parse()

    if args.testset is not None:
        testsets = args.testset
    else:
        testsets = range(0, num_subsets)

    # for testset in testsets:
    # # for testset in range(0, 1):
    #     training_filepath = os.path.join(args.sampled_filelist_path, 'training_filenames_testset{}.npy'.format(testset))
    #     training_filepath = training_filepath.replace('\\', '/')
    #     validation_filepath = os.path.join(args.sampled_filelist_path, 'validation_filenames_testset{}.npy'.format(testset))
    #     validation_filepath = validation_filepath.replace('\\', '/')
    #     test_filepath = os.path.join(args.sampled_filelist_path, 'test_filenames_testset{}.npy'.format(testset))
    #     test_filepath = test_filepath.replace('\\', '/')
    #     if regenerate_file_list:
    #         training_filenames, validation_filenames, test_filenames = prepare_input_filenames(
    #             testset=str(testset), num_subsets=num_subsets)
    #         np.save(training_filepath, training_filenames)
    #         np.save(validation_filepath, validation_filenames)
    #         np.save(test_filepath, test_filenames)

    AUC = []

    plt.figure()
    plt.plot([0, 1], [0, 1], 'k--')
    for testset in testsets:
        '''
        Prepare input file names
        '''
        training_filepath = os.path.join(args.sampled_filelist_path, 'training_filenames_testset{}.npy'.format(testset))
        training_filepath = training_filepath.replace('\\', '/')
        validation_filepath = os.path.join(args.sampled_filelist_path, 'validation_filenames_testset{}.npy'.format(testset))
        validation_filepath = validation_filepath.replace('\\', '/')
        # test_filepath = os.path.join(sampled_filelist_path, 'test_filenames_testset{}.npy'.format(testset))
        test_filepath = os.path.join(args.sampled_filelist_path, 'test{}_filenames_testset{}.npy'.format(times, testset))
        test_filepath = test_filepath.replace('\\', '/')
        training_filenames = np.load(training_filepath)
        validation_filenames = np.load(validation_filepath)
        test_filenames = np.load(test_filepath)
        print("training size : ", len(training_filenames), "validation size: ", len(validation_filenames),
              "test size: ", len(test_filenames))

        '''
        Execute various strategies
        '''

        # print("1  mixed_3channels_2d CNN")
        # sys.stdout.flush()
        pickle_path = args.result_path_base_new + 'method_{}_testset{}.pickle'.format(args.method, testset)
        if os.path.isfile(pickle_path):
            results = pickle.load(open(pickle_path, 'rb'))
        else:
            result_csv_path = args.result_path_base_new + 'method_{}_testset{}.csv'.format(args.method, testset)
            model_path = args.result_path_base + 'model_{}_testset{}last.pth.tar'.format(args.method, testset)
            # model_path = args.result_path_base + 'model_{}_testset{}last.pth.tar'.format(args.method, testset)
            results = training_loop(args, training_filenames, validation_filenames, test_filenames,
                                    model_path, result_csv_path, method=args.method)
            with open(pickle_path, 'wb') as f:
                pickle.dump(results, f)
        # plot_auc(results[0], results[1], results[2], 'ROC_{}_testset{}'.format(args.method, testset))
        # plt.figure()
        # plt.plot([0, 1], [0, 1], 'k--')
        # plt.plot(results[0], results[1], label='{} CNN (area = {:.3f})'.format(args.method, results[2]))
        plt.plot(results[0], results[1], label='testset{} (area = {:.3f})'.format(testset, results[2]))
        plt.xlabel('False positive rate')
        plt.ylabel('True positive rate')
        plt.title('ROC curve {}'.format(args.method))
        plt.legend(loc='best')
        # plt.show()
        # plt.savefig(args.result_path_base + 'strategy_comparison_vgg_testset{}.png'.format(testset))
        plt.savefig(args.result_path_base_new + 'ROC_{}.png'.format(args.method))
        print("Test subset: ", testset)
        print("results: {}".format(args.method), results[2:])
        # plt.close('all')
        AUC.append(results[2])

    with open(os.path.join(args.result_path_base_new, 'AUC_{}.txt'.format(args.method)), "w") as f:
        f.write("AUC {}: {}. Average: {}\n".format(args.method, AUC, sum(AUC) / float(len(AUC))))
