from facelib import FaceType

# Experiment name
exp_name = "exp"
pretrained_model_path = None

# Dataloader
# Input aligned face dataset path
src_aligned_path = "/home/tunm/datasets/deepfacelab_datasets/src_aligned"
dst_aligned_path = "/home/tunm/datasets/deepfacelab_datasets/dst_aligned"

# Data augmentation
is_debug = False
batch_size = 16
data_format = "NCHW"
random_ct_samples_path = None
random_src_flip = True
random_dst_flip = True
random_warp = True
ct_mode = None
random_hsv_power = 0.0
face_type = FaceType.WHOLE_FACE
uniform_yaw = True
src_generators_count = 8
dst_generators_count = 8
src_pak_name = None
dst_pak_name = None
ignore_same_path = False
scale_range = [-0.15, 0.15]
random_downsample = False
random_noise = False
random_blur = False
random_jpeg = False
random_shadow = False
num_workers = 4

# Model
opts = ["u", "d", "t"]
resolution = 256
input_channels = 3
e_dims = 64
ae_dims = 512
d_dims = 64
d_mask_dims = 32

# Loss
ssim_weight = 5
seg_mse_weight = 10
mask_mse_weight = 10
part_prio_weight = 300

# Training
visualize_once_flag = True
total_iter = 800000
lr = 3e-5
lr_cos = 500
lr_dropout = False
fp16 = False
warmup_iter = 0
clip_grad = 0.0
grad_accum_steps = 1
masked_training = True
blur_out_mask = True
use_part_loss = True
save_model_interval = 2000
draw_results_interval = 100
log_interval = 100
show_grid_num = 10

# Test
test_fixed_samples_path = None
test_fixed_samples_batch_size = 4