import numpy as np

import os
import sys
import random
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


def random_downsample_candidates(filenames):
    positive_candidates_plus_aug = [s for s in filenames if "real_" in s]
    positive_candidates = [s for s in positive_candidates_plus_aug if "random" not in s]
    """positive_aug_10 = [s for s in positive_candidates_plus_aug if "real_size64x64_random10" in s]
    sampled_positive_candidates = np.append(positive_candidates, positive_aug_10)"""
    positive_aug = [s for s in positive_candidates_plus_aug if "real_size64x64_random" in s]
    sampled_positive_candidates = np.append(positive_candidates, positive_aug)

    negative_candidates = [s for s in filenames if "fake_" in s]
    print("Number of positive candidates in subset: ", len(positive_candidates))
    print("Number of negative candidates in subset: ", len(negative_candidates))
    # numbers = random.sample(range(0, len(negative_candidates)), len(sampled_positive_candidates))
    numbers = random.sample(range(0, len(negative_candidates)), len(sampled_positive_candidates) if len(negative_candidates) > len(sampled_positive_candidates) else len(negative_candidates))
    numbers.sort()
    print("length of numbers: ", len(numbers), type(numbers), numbers[0:10])

    file_list = np.append(sampled_positive_candidates, np.array(negative_candidates)[numbers])
    np.random.shuffle(file_list)

    print("Length of samples in subset: ", len(file_list))
    return file_list


def downsample_candidates(filenames):
    positive_candidates = [s for s in filenames if "real_size64x64.npy" in s]
    negative_candidates = [s for s in filenames if "fake_size64x64.npy" in s]
    print("Number of positive candidates in subset: ", len(positive_candidates))
    print("Number of negative candidates in subset: ", len(negative_candidates))
    numbers = random.sample(range(0, len(negative_candidates)), len(positive_candidates))
    numbers.sort()
    print("Length of numbers: ", len(numbers), type(numbers), numbers[0:10])
    
    sampled_filenames = np.array([])
    count = 0
    for i, ID in enumerate(filenames):
        if ID in positive_candidates:
            sampled_filenames = np.append(sampled_filenames, ID)
        elif 'fake_' in  ID:
            if count in numbers:
                sampled_filenames = np.append(sampled_filenames, ID)
            count = count + 1
    print("Number of candidates after downsampling: ", len(sampled_filenames))
    return sampled_filenames


def get_all_candidate_filename(path):
    list_real = search(path, 'real_size64x64')
    list_fake = search(path, 'fake_size64x64')
    return list_real+list_fake


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


if __name__ == '__main__':
    #num_subsets = 10
    if len(sys.argv) == 2:
        subset = sys.argv[1]
        print("saving file names for subset %s " % subset)
        #sys.stdout.flush()
        normalization_output_path = normalized_volume_path + 'subset%s' % subset
        all_filenames_in_this_subset = get_all_candidate_filename(normalization_output_path)
        np.save(os.path.join(normalized_filelist_path, 'subset%s_filenames.npy' % subset), all_filenames_in_this_subset)
        print("subset %s finished." % subset)
    else:
        for i in range(0, num_subsets):
            normalization_output_path = normalized_volume_path + 'subset' + str(i)
            all_filenames_in_this_subset = get_all_candidate_filename(normalization_output_path)

            ''' Full file name list'''
            print("saving file names for subset " + str(i))
            #sys.stdout.flush()
            np.save(os.path.join(normalized_filelist_path, 'subset' + str(i) + '_filenames.npy'), all_filenames_in_this_subset)

            '''Sampled filename list'''
            # all_filenames_in_this_subset = np.load(os.path.join(full_filename_list_path, 'subset' + str(i) + '_filenames.npy'))
            sampled_filenames = random_downsample_candidates(all_filenames_in_this_subset)
            if not os.path.exists(sampled_filelist_path):
                os.mkdir(sampled_filelist_path)
            np.save(os.path.join(sampled_filelist_path, 'subset' + str(i) + '_filenames.npy'), sampled_filenames)

            print("subset " + str(i) + " finished.")
            #sys.stdout.flush()
    print("Successfully finished\n")
