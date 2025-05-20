from __future__ import division
import glob
import pandas as pd
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


def concat_files(submission_path, method):
    # submission_paths = glob.glob(submission_path + '/models_3channelsx18aug/' + method + '_testset*.csv')
    submission_paths = glob.glob(submission_path + method + '_testset*.csv')
    combined_csv = pd.concat([pd.read_csv(f) for f in submission_paths])
    return combined_csv


if __name__ == "__main__":
    result_path_base = '/Path/to/Result/'
    submission_path_first_part = result_path_base
    method_prefix_list = [
        'method_2D',
        'method_2D_TL',
        'method_2.5D',
        'method_2.5D_TL',
        'method_3D',
        'method_2.75D',
        'method_2.75D_TL',
        'method_2.75D_3channel',
        'method_2.75D_3channel_TL',
    ]
    for method_prefix in method_prefix_list:
        file = concat_files(submission_path_first_part, method_prefix)
        file.to_csv(result_path_base + method_prefix + '_ensemble.csv',
                    columns=['seriesuid', 'coordX', 'coordY', 'coordZ', 'probability'])

        print(method_prefix, 'finished')
    print('finished all')
