# -*- coding: utf-8 -*-
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
    if not (reuse_results and os.path.isfile(result_path_base + label + '.png')):
        plt.savefig(result_path_base + label + '.png')


if __name__ == '__main__':
    # import tensorflow as tf

    # # tf.debugging.set_log_device_placement(True)
    # # print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
    # gpu_devices = tf.config.experimental.list_physical_devices('GPU')
    # for device in gpu_devices:
    #     tf.config.experimental.set_memory_growth(device, True)

    # plt.ion()

    parser = argparse.ArgumentParser(description='Run nodule classification on various data split')
    parser.add_argument('--testset', type=int, nargs='+', help='choose test set(s)')

    args = parser.parse_args()
    print(args)

    if args.testset is not None:
        testsets = args.testset
    else:
        testsets = range(0, num_subsets)

    matrices = ['AUC', 'ACC', 'Sensitivity', 'Specificity', 'Precision', 'Recall', 'F1_score']
    methods = [
        # 'singleview_2d',
               'vgg_singleview',
               # 'multiview_2d_late',
               'vgg_multiview_2d_late',
               'volume_3d',
               # 'spiral',
               'vgg',
               # 'spiral_3channels',
               'spiral_3channels_vgg',
               # 'singleview_mixed_2d',
        'singleview_mixed_2d_vgg',
               # 'mixed_multiview_2d_late',
               'mixed_multiview_2d_late_vgg',
               '_mixed_3channels_2D',
               '_mixed_3channels_2D_vgg',
               '_mixed_3channels_2.5D',
               '_mixed_3channels_2.5D_vgg',
               ]
    result = {}

    for matrix in matrices:
        for method in methods:
            dict_add = {'{}_{}'.format(matrix, method): []}
            result.update(dict_add)


    for testset in testsets:
        plt.figure()
        plt.plot([0, 1], [0, 1], 'k--')

        for method in methods:
            print('{} CNN'.format(method))
            pickle_path = result_path_base + '{}_testset{}.pickle'.format(method, testset)
            if os.path.isfile(pickle_path):
                results = pickle.load(open(pickle_path, 'rb'))

                # plot_auc(results[0], results[1], results[2],
                #          'roc_{}_testset' + str(method, testset))

                plt.plot(results[0], results[1], label='{} CNN (area = {:.3f})'.format(method, results[2]))

                result['AUC_{}'.format(method)].append(results[2])
                result['ACC_{}'.format(method)].append(results[3])
                result['Sensitivity_{}'.format(method)].append(results[4])
                result['Specificity_{}'.format(method)].append(results[5])
                result['Precision_{}'.format(method)].append(results[6])
                result['Recall_{}'.format(method)].append(results[7])
                result['F1_score_{}'.format(method)].append(results[8])

        plt.xlabel('False positive rate')
        plt.ylabel('True positive rate')
        plt.title('ROC curve')
        plt.legend(loc='best')
        # plt.show()
        plt.savefig(result_path_base + 'strategy_comparison_vgg_testset{}.png'.format(testset))


        plt.close('all')




    with open(os.path.join(result_path_base, 'Classification_Result{}_overall.txt'.format(times)), "w") as f:
        for matrix in matrices:
            for method in methods:
                f.write("{} {} Average: {}\n".format(matrix, method, sum(result['{}_{}'.format(matrix, method)]) /
                                                      float(len(result['{}_{}'.format(matrix, method)]))))

        # f.write("AUC 2D Average: {}\n".format(sum(AUC_singleview_2d) /
        #                                       float(len(AUC_singleview_2d))))
        # f.write("AUC 2D + TL Average: {}\n".format(sum(AUC_vgg_singleview) /
        #                                            float(len(AUC_vgg_singleview))))
        # f.write("AUC 2.5D Average: {}\n".format(sum(AUC_multiview_2d_late) /
        #                                         float(len(AUC_multiview_2d_late))))
        # f.write("AUC 2.5D + TL Average: {}\n".format(sum(AUC_multiview_2d_late_vgg) /
        #                                              float(len(AUC_multiview_2d_late_vgg))))
        # f.write("AUC 3D Average: {}\n".format(sum(AUC_3D) /
        #                                       float(len(AUC_3D))))
        # f.write("AUC 2.75D Average: {}\n".format(sum(AUC_spiral) /
        #                                          float(len(AUC_spiral))))
        # f.write("AUC 2.75D + T Average: {}\n".format(sum(AUC_vgg) /
        #                                              float(len(AUC_vgg))))
        # f.write("AUC 2.75D 3 channels Average: {}\n".format(sum(AUC_spiral_3channels) /
        #                                                     float(len(AUC_spiral_3channels))))
        # f.write("AUC 2.75D 3 channels + TL Average: {}\n".format(sum(AUC_spiral_3channels_vgg) /
        #                                                          float(len(AUC_spiral_3channels_vgg))))
        #
        # f.write("ACC 2D Average: {}\n".format(sum(ACC_singleview_2d) /
        #                                       float(len(ACC_singleview_2d))))
        # f.write("ACC 2D + TL Average: {}\n".format(sum(ACC_vgg_singleview) /
        #                                            float(len(ACC_vgg_singleview))))
        # f.write("ACC 2.5D Average: {}\n".format(sum(ACC_multiview_2d_late) /
        #                                         float(len(ACC_multiview_2d_late))))
        # f.write("ACC 2.5D + TL Average: {}\n".format(sum(ACC_multiview_2d_late_vgg) /
        #                                              float(len(ACC_multiview_2d_late_vgg))))
        # f.write("ACC 3D Average: {}\n".format(sum(ACC_3D) /
        #                                       float(len(ACC_3D))))
        # f.write("ACC 2.75D Average: {}\n".format(sum(ACC_spiral) /
        #                                          float(len(ACC_spiral))))
        # f.write("ACC 2.75D + T Average: {}\n".format(sum(ACC_vgg) /
        #                                              float(len(ACC_vgg))))
        # f.write("ACC 2.75D 3 channels Average: {}\n".format(sum(ACC_spiral_3channels) /
        #                                                     float(len(ACC_spiral_3channels))))
        # f.write("ACC 2.75D 3 channels + TL Average: {}\n".format(sum(ACC_spiral_3channels_vgg) /
        #                                                          float(len(ACC_spiral_3channels_vgg))))
        #
        # f.write("Sensitivity 2D Average: {}\n".format(sum(Sensitivity_singleview_2d) /
        #                                       float(len(Sensitivity_singleview_2d))))
        # f.write("Sensitivity 2D + TL Average: {}\n".format(sum(Sensitivity_vgg_singleview) /
        #                                            float(len(Sensitivity_vgg_singleview))))
        # f.write("Sensitivity 2.5D Average: {}\n".format(sum(Sensitivity_multiview_2d_late) /
        #                                         float(len(Sensitivity_multiview_2d_late))))
        # f.write("Sensitivity 2.5D + TL Average: {}\n".format(sum(Sensitivity_multiview_2d_late_vgg) /
        #                                              float(len(Sensitivity_multiview_2d_late_vgg))))
        # f.write("Sensitivity 3D Average: {}\n".format(sum(Sensitivity_3D) /
        #                                       float(len(Sensitivity_3D))))
        # f.write("Sensitivity 2.75D Average: {}\n".format(sum(Sensitivity_spiral) /
        #                                          float(len(Sensitivity_spiral))))
        # f.write("Sensitivity 2.75D + T Average: {}\n".format(sum(Sensitivity_vgg) /
        #                                              float(len(Sensitivity_vgg))))
        # f.write("Sensitivity 2.75D 3 channels Average: {}\n".format(sum(Sensitivity_spiral_3channels) /
        #                                                     float(len(Sensitivity_spiral_3channels))))
        # f.write("Sensitivity 2.75D 3 channels + TL Average: {}\n".format(sum(Sensitivity_spiral_3channels_vgg) /
        #                                                          float(len(Sensitivity_spiral_3channels_vgg))))
        #
        # f.write("Specificity 2D Average: {}\n".format(sum(Specificity_singleview_2d) /
        #                                               float(len(Specificity_singleview_2d))))
        # f.write("Specificity 2D + TL Average: {}\n".format(sum(Specificity_vgg_singleview) /
        #                                                    float(len(Specificity_vgg_singleview))))
        # f.write("Specificity 2.5D Average: {}\n".format(sum(Specificity_multiview_2d_late) /
        #                                                 float(len(Specificity_multiview_2d_late))))
        # f.write("Specificity 2.5D + TL Average: {}\n".format(sum(Specificity_multiview_2d_late_vgg) /
        #                                                      float(len(Specificity_multiview_2d_late_vgg))))
        # f.write("Specificity 3D Average: {}\n".format(sum(Specificity_3D) /
        #                                               float(len(Specificity_3D))))
        # f.write("Specificity 2.75D Average: {}\n".format(sum(Specificity_spiral) /
        #                                                  float(len(Specificity_spiral))))
        # f.write("Specificity 2.75D + T Average: {}\n".format(sum(Specificity_vgg) /
        #                                                      float(len(Specificity_vgg))))
        # f.write("Specificity 2.75D 3 channels Average: {}\n".format(sum(Specificity_spiral_3channels) /
        #                                                             float(len(Specificity_spiral_3channels))))
        # f.write("Specificity 2.75D 3 channels + TL Average: {}\n".format(sum(Specificity_spiral_3channels_vgg) /
        #                                                                  float(len(Specificity_spiral_3channels_vgg))))
        #
        # f.write("Precision 2D Average: {}\n".format(sum(Precision_singleview_2d) /
        #                                               float(len(Precision_singleview_2d))))
        # f.write("Precision 2D + TL Average: {}\n".format(sum(Precision_vgg_singleview) /
        #                                                    float(len(Precision_vgg_singleview))))
        # f.write("Precision 2.5D Average: {}\n".format(sum(Precision_multiview_2d_late) /
        #                                                 float(len(Precision_multiview_2d_late))))
        # f.write("Precision 2.5D + TL Average: {}\n".format(sum(Precision_multiview_2d_late_vgg) /
        #                                                      float(len(Precision_multiview_2d_late_vgg))))
        # f.write("Precision 3D Average: {}\n".format(sum(Precision_3D) /
        #                                               float(len(Precision_3D))))
        # f.write("Precision 2.75D Average: {}\n".format(sum(Precision_spiral) /
        #                                                  float(len(Precision_spiral))))
        # f.write("Precision 2.75D + T Average: {}\n".format(sum(Precision_vgg) /
        #                                                      float(len(Precision_vgg))))
        # f.write("Precision 2.75D 3 channels Average: {}\n".format(sum(Precision_spiral_3channels) /
        #                                                             float(len(Precision_spiral_3channels))))
        # f.write("Precision 2.75D 3 channels + TL Average: {}\n".format(sum(Precision_spiral_3channels_vgg) /
        #                                                                  float(len(Precision_spiral_3channels_vgg))))
        #
        # f.write("Recall 2D Average: {}\n".format(sum(Recall_singleview_2d) /
        #                                               float(len(Recall_singleview_2d))))
        # f.write("Recall 2D + TL Average: {}\n".format(sum(Recall_vgg_singleview) /
        #                                                    float(len(Recall_vgg_singleview))))
        # f.write("Recall 2.5D Average: {}\n".format(sum(Recall_multiview_2d_late) /
        #                                                 float(len(Recall_multiview_2d_late))))
        # f.write("Recall 2.5D + TL Average: {}\n".format(sum(Recall_multiview_2d_late_vgg) /
        #                                                      float(len(Recall_multiview_2d_late_vgg))))
        # f.write("Recall 3D Average: {}\n".format(sum(Recall_3D) /
        #                                               float(len(Recall_3D))))
        # f.write("Recall 2.75D Average: {}\n".format(sum(Recall_spiral) /
        #                                                  float(len(Recall_spiral))))
        # f.write("Recall 2.75D + T Average: {}\n".format(sum(Recall_vgg) /
        #                                                      float(len(Recall_vgg))))
        # f.write("Recall 2.75D 3 channels Average: {}\n".format(sum(Recall_spiral_3channels) /
        #                                                             float(len(Recall_spiral_3channels))))
        # f.write("Recall 2.75D 3 channels + TL Average: {}\n".format(sum(Recall_spiral_3channels_vgg) /
        #                                                                  float(len(Recall_spiral_3channels_vgg))))
        #
        # f.write("F1_score 2D Average: {}\n".format(sum(F1_score_singleview_2d) /
        #                                               float(len(F1_score_singleview_2d))))
        # f.write("F1_score 2D + TL Average: {}\n".format(sum(F1_score_vgg_singleview) /
        #                                                    float(len(F1_score_vgg_singleview))))
        # f.write("F1_score 2.5D Average: {}\n".format(sum(F1_score_multiview_2d_late) /
        #                                                 float(len(F1_score_multiview_2d_late))))
        # f.write("F1_score 2.5D + TL Average: {}\n".format(sum(F1_score_multiview_2d_late_vgg) /
        #                                                      float(len(F1_score_multiview_2d_late_vgg))))
        # f.write("F1_score 3D Average: {}\n".format(sum(F1_score_3D) /
        #                                               float(len(F1_score_3D))))
        # f.write("F1_score 2.75D Average: {}\n".format(sum(F1_score_spiral) /
        #                                                  float(len(F1_score_spiral))))
        # f.write("F1_score 2.75D + T Average: {}\n".format(sum(F1_score_vgg) /
        #                                                      float(len(F1_score_vgg))))
        # f.write("F1_score 2.75D 3 channels Average: {}\n".format(sum(F1_score_spiral_3channels) /
        #                                                             float(len(F1_score_spiral_3channels))))
        # f.write("F1_score 2.75D 3 channels + TL Average: {}\n".format(sum(F1_score_spiral_3channels_vgg) /
        #                                                                  float(len(F1_score_spiral_3channels_vgg))))