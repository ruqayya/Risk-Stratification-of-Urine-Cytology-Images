import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# dataset details
save_intermediate_results = 0   # this will save the patches along with their segmentation mask
all_class_names = ['normal', 'squamous', 'inflammatory', 'others', 'atypia', 'malignant', 'debri']

# Network related parameters
batch_size = 20
network_patch_size = 256
learning_rate = 1e-05

# Setting paths
current_dir = os.path.dirname(__file__)
checkpoint_dir = os.path.join(current_dir, 'checkpoints', 'model_weights-0.83-328.hdf5')

wsi_dir = os.path.join(current_dir, 'WSIs')
output_path = os.path.join(current_dir, 'output')
os.makedirs(output_path, exist_ok=True)
