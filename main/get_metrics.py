# -*- coding: utf-8 -*-
import argparse
import pickle
import re
import numpy as np
import matplotlib.pyplot as plt
import sys
# print(sys.path)
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

# os.environ["CUDA_VISIBLE_DEVICES"] = "6"


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


def arg_parse():
    parser = argparse.ArgumentParser(description='Run nodule classification on various data split')
    parser.add_argument('--testset', type=int, nargs='+',
                        help='choose test set(s)')
    parser.add_argument('--result_path_base', type=str,
                        default='/home/x.wang/new_nki_project/2.75D-pytorch_luna/Result/result_down_ratio1_org_last/',
                        help='')

    args = parser.parse_args()


    print(args)
    return args

if __name__ == '__main__':
    # import tensorflow as tf

    # # tf.debugging.set_log_device_placement(True)
    # # print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
    # gpu_devices = tf.config.experimental.list_physical_devices('GPU')
    # for device in gpu_devices:
    #     tf.config.experimental.set_memory_growth(device, True)

    # plt.ion()

    # parser = argparse.ArgumentParser(description='Run nodule classification on various data split')
    # parser.add_argument('--testset', type=int, nargs='+', help='choose test set(s)')

    args = arg_parse()
    # print(args)

    if args.testset is not None:
        testsets = args.testset
    else:
        testsets = range(0, num_subsets)

    matrices = ['AUC',
                'ACC', 'Sensitivity', 'Specificity', 'Precision', 'Recall', 'F1_score'
                ]
    methods = [
        '2D',
        '2D_TL',
        '2.5D',
        '2.5D_TL',
        '3D',
        '2.75D',
        '2.75D_TL',
        '2.75D_3channel',
        '2.75D_3channel_TL',
    ]
    result = {}

    for matrix in matrices:
        for method in methods:
            dict_add = {'{}_{}'.format(matrix, method): []}
            result.update(dict_add)


    for testset in testsets:
        # plt.figure()
        # plt.plot([0, 1], [0, 1], 'k--')

        for method in methods:
            print('{} CNN'.format(method))
            pickle_path = args.result_path_base + 'method_{}_testset{}.pickle'.format(method, testset)
            if os.path.isfile(pickle_path):
                results = pickle.load(open(pickle_path, 'rb'))

                # plot_auc(results[0], results[1], results[2],
                #          'roc_{}_testset' + str(method, testset))

                # plt.plot(results[0], results[1], label='{} CNN (area = {:.3f})'.format(method, results[2]))

                result['AUC_{}'.format(method)].append(results[2])
                result['ACC_{}'.format(method)].append(results[3])
                result['Sensitivity_{}'.format(method)].append(results[4])
                result['Specificity_{}'.format(method)].append(results[5])
                result['Precision_{}'.format(method)].append(results[6])
                result['Recall_{}'.format(method)].append(results[7])
                result['F1_score_{}'.format(method)].append(results[8])

        # plt.xlabel('False positive rate')
        # plt.ylabel('True positive rate')
        # plt.title('ROC curve')
        # plt.legend(loc='best')
        # plt.show()
        # plt.savefig(args.result_path_base + 'strategy_comparison_vgg_testset{}.png'.format(testset))


        # plt.close('all')




    with open(os.path.join(args.result_path_base, 'Classification_Result{}_overall.txt'.format(times)), "w") as f:
        for matrix in matrices:
            for method in methods:
                f.write("{} {} Average: {}\n".format(matrix, method, sum(result['{}_{}'.format(matrix, method)]) /
                                                      float(len(result['{}_{}'.format(matrix, method)]))))
