import os
import platform
from pathlib import Path

''' Input directory and files '''
if platform.system() == "Windows":
    base_dir = "XXX"  # windows
else:
    base_dir = "XXX"  # linux

candidate_file = base_dir + 'candidates_V2.csv'
use_full_dataset = True
reuse_model = False
reuse_results = True
regenerate_file_list = False
deep = False
training_data_downsample_ratio = 2
num_subsets = 10
times = ''

''' Input directories '''
plot_output_path_base = base_dir + 'cubic_npy/'
Path(plot_output_path_base).mkdir(parents=True, exist_ok=True)


normalized_volume_path = base_dir + 'normalized_cubic_npy/'
Path(normalized_volume_path).mkdir(parents=True, exist_ok=True)
normalized_filelist_path = normalized_volume_path + 'filenames'
Path(normalized_filelist_path).mkdir(parents=True, exist_ok=True)


error_candidates_path_base = base_dir + 'error_candidates/'
Path(error_candidates_path_base).mkdir(parents=True, exist_ok=True)
result_path_base = base_dir + 'results/'
os.makedirs(result_path_base, exist_ok=True)
sampled_filelist_path = result_path_base + 'filenames'
Path(sampled_filelist_path).mkdir(parents=True, exist_ok=True)

'''  training and test configuration '''

use_gpu=True

learning_rate = 0.00001  # 0.00001  # 0.00001 for deep 3d cnn
learning_rate_single_view = 0.00001
learning_rate_multi_view = 0.00001
learning_rate_spiral = 0.00001
learning_rate_3d = 0.00001
learning_rate_vgg = 0.00001
batch_size = 16  #16
n_classes = 2
n_channels = 1
n_channels_3spiral = 3
n_channels_multiview = 9
num_epochs = 50  #50
num_works = 8

patient_epochs = 50
# train_augmentation = True
train_augmentation = False
spiral_augmentation = 'none'
