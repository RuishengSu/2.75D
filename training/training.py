# -*- coding: utf-8 -*-

from sklearn.metrics import roc_curve, auc, confusion_matrix, classification_report, \
    recall_score, precision_score, f1_score
from data.get_all_filenames import *
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
import csv
import re
import numpy as np

import os
import torch
from data.data_generators import dataloader, get_labels
from .train_val_test_demo import train_val_test_demo

from models.models import SingleView_model, Spiral_3channels_model


def get_model(method, deep):
    if method == '2.75D':
        model = SingleView_model(transfer_learning=False, Spiral_Img=True, fusion_method='committee', deep=deep)
    elif method == '2.75D_TL':
        model = SingleView_model(transfer_learning=True, Spiral_Img=True, fusion_method='committee', deep=deep)
    elif method == '2.75D_3channel':
        model = Spiral_3channels_model(transfer_learning=False, Spiral_Img=True, fusion_method='late', deep=deep)
    elif method == '2.75D_3channel_TL':
        model = Spiral_3channels_model(transfer_learning=True, Spiral_Img=True, fusion_method='late', deep=deep)
    else:
        raise Exception('Method {} is not supported'.format(method))

    return model

def training_loop(args, training_filenames, validation_filenames, test_filenames, model_path, result_csv_path, method='2.75D'):
    test_params_singleview = {
        'method': method,
        'shuffle': False,
        'augmentation': False,
        'batch_size': 1,
        'num_workers': num_works
    }

    # print("Backend: ", backend.backend())
    #print("Using GPU or CPU: ", backend.tensorflow_backend._get_available_gpus())

    '''
    Start training process.
    '''

    # training_dataloader = dataloader(training_filenames, **train_params_singleview)
    # validation_dataloader = dataloader(validation_filenames, **validation_params_singleview)
    '''Model'''
    model = get_model(method, deep)
    print("Model created")
    print(model)
    demo = train_val_test_demo(model, model_path, method, epochs=num_epochs, lr=learning_rate, use_gpu=use_gpu)
    demo.load_model(model_path)

    '''
    Score trained model.
    '''
    test_dataloader = dataloader(test_filenames, **test_params_singleview)
    #Show ROC and AUC
    scores = demo.test(test_dataloader)
    scores = scores.squeeze()

    """scores = model.predict_generator(test_generator,
                                     workers=multiprocessing.cpu_count(),
                                     use_multiprocessing=True)"""

    # for each image in the testing set we need to find the index of the
    # label with corresponding largest predicted probability

    y_test, seriesuids, coordXs, coordYs, coordZs = get_labels(test_filenames)

    y_pred_idxs = np.argmax(scores, axis=1)
    print('scores shape', scores.shape)
    print("y_pred_score: ", y_pred_idxs, "size: ", np.shape(y_pred_idxs))
    # # show a nicely formatted classification report
    # report = classification_report(y_test, y_pred_idxs, target_names=['class 0: non-nodule', 'class 1: nodule'])
    # print(report)
    # result_report_path = result_csv_path
    # result_report_path = result_report_path.replace('.csv', 'report.txt')
    # text_file = open(result_report_path, "w")
    # text_file.write(report)
    # text_file.close()
    y_pred_probs = scores[:, 1]

    with open(result_csv_path, mode='w') as result_csv_file:
        writer = csv.writer(result_csv_file, delimiter=',')
        writer.writerow(['seriesuid', 'coordX', 'coordY', 'coordZ', 'probability'])
        for ln, seriesuid in enumerate(seriesuids):
            writer.writerow([seriesuid, coordXs[ln], coordYs[ln], coordZs[ln], y_pred_probs[ln]])

    '''
    Prepare outputs/results
    '''
    # compute the confusion matrix and and use it to derive the raw
    # accuracy, sensitivity, and specificity
    cm = confusion_matrix(y_test, y_pred_idxs)
    total = sum(sum(cm))
    acc = (cm[0, 0] + cm[1, 1]) / total
    sensitivity = cm[0, 0] / (cm[0, 0] + cm[0, 1])
    specificity = cm[1, 1] / (cm[1, 0] + cm[1, 1])

    precision = precision_score(y_test, y_pred_idxs)
    recall = recall_score(y_test, y_pred_idxs)
    f1 = f1_score(y_test, y_pred_idxs)

    # show the confusion matrix, accuracy, sensitivity, and specificity
    print(cm)
    print("acc: {:.4f}".format(acc))
    print("sensitivity: {:.4f}".format(sensitivity))
    print("specificity: {:.4f}".format(specificity))

    print("precision: {:.4f}".format(precision))
    print("recall: {:.4f}".format(recall))
    print("F1-Score: {:.4f}".format(f1))

    fpr, tpr, thresholds = roc_curve(y_test, y_pred_probs)
    auc_score = auc(fpr, tpr)
    print("fpr: ", fpr)
    print("tpr: ", tpr)
    print("auc_score: ", auc_score)
    sys.stdout.flush()
    return [fpr, tpr, auc_score, acc, sensitivity, specificity, precision, recall, f1]
