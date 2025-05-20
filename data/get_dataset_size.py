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
import numpy as np


def get_dataset_sizes():
    total_nodule = 0
    total_non_nodule = 0
    for subset_index in range(0, 10):
        # subset_path = os.path.join(full_filename_list_path, 'subset' + str(subset_index) + '_filenames.npy')
        subset_path = os.path.join(sampled_filelist_path, 'subset' + str(subset_index) + '_filenames.npy')
        sample_filenames = np.load(subset_path)
        #print(sample_filenames)
        positive_candidates = [s for s in sample_filenames if "real_size64x64.npy" in s]
        negative_candidates = [s for s in sample_filenames if "fake_size64x64.npy" in s]
        augmented_candidates = [s for s in sample_filenames if "random" in s]

        nr_pos = len(positive_candidates)
        nr_neg = len(negative_candidates)
        nr_aug = len(augmented_candidates)

        print("Subset ", subset_index, "----- [ real: ", nr_pos, "aug: ", nr_aug,
              " real+aug: ", nr_pos + nr_aug, " fake: ", nr_neg, " total: ", len(sample_filenames), " ]")
        total_nodule = total_nodule + nr_pos
        total_non_nodule = total_non_nodule + nr_neg
    print("-----------------------------------------")
    print("Total number of nodules: ", total_nodule)
    print("Total number of non nodules: ", total_non_nodule)
    print("-----------------------------------------")


if __name__ == '__main__':
    get_dataset_sizes()

