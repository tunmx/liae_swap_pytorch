from pathlib import Path
import torch
from torch.utils.data import Dataset
from facelib import FaceType
from samplelib import *


class FaceSwapDataset(Dataset):
    
    def __init__(self, 
                 data_path,
                 dataset_type='src', 
                 random_ct_samples_path=None,
                 is_debug=False,
                 random_flip=True,
                 random_warp=True,
                 ct_mode=None,
                 random_hsv_power=0.0,
                 face_type=FaceType.WHOLE_FACE,
                 data_format="NCHW",
                 resolution=256,
                 uniform_yaw_distribution=False,
                 generators_count=8,
                 pak_name=None,
                 ignore_same_path=False,
                 scale_range=[-0.15, 0.15],
                 random_downsample=False,
                 random_noise=False,
                 random_blur=False,
                 random_jpeg=False,
                 random_shadow=False,
                 channel_type=SampleProcessor.ChannelType.BGR,
                 max_samples=100000000):
        self.data_path = Path(data_path)
        self.dataset_type = dataset_type
        self.max_samples = max_samples
        
        self.generator = SampleGeneratorFace(
            self.data_path,
            pak_name=pak_name,
            ignore_same_path=ignore_same_path,
            random_ct_samples_path=random_ct_samples_path,
            debug=is_debug,
            batch_size=1,
            sample_process_options=SampleProcessor.Options(
                scale_range=scale_range, 
                random_flip=random_flip
            ),
            output_sample_types=[
                {
                    "sample_type": SampleProcessor.SampleType.FACE_IMAGE,
                    "warp": random_warp,
                    "random_downsample": random_downsample,
                    "random_noise": random_noise,
                    "random_blur": random_blur,
                    "random_jpeg": random_jpeg,
                    "random_shadow": random_shadow,
                    "random_hsv_shift_amount": random_hsv_power,
                    "transform": True,
                    "channel_type": channel_type,
                    "ct_mode": ct_mode,
                    "face_type": face_type,
                    "data_format": data_format,
                    "resolution": resolution,
                },
                {
                    "sample_type": SampleProcessor.SampleType.FACE_IMAGE,
                    "warp": False,
                    "transform": True,
                    "channel_type": channel_type,
                    "ct_mode": ct_mode,
                    "random_hsv_shift_amount": random_hsv_power,
                    "random_shadow": random_shadow,
                    "face_type": face_type,
                    "data_format": data_format,
                    "resolution": resolution,
                },
                {
                    "sample_type": SampleProcessor.SampleType.FACE_MASK,
                    "warp": False,
                    "transform": True,
                    "channel_type": SampleProcessor.ChannelType.G,
                    "face_mask_type": SampleProcessor.FaceMaskType.FULL_FACE,
                    "face_type": face_type,
                    "data_format": data_format,
                    "resolution": resolution,
                },
                {
                    "sample_type": SampleProcessor.SampleType.FACE_MASK,
                    "warp": False,
                    "transform": True,
                    "channel_type": SampleProcessor.ChannelType.G,
                    "face_mask_type": SampleProcessor.FaceMaskType.FULL_FACE_EYES,
                    "face_type": face_type,
                    "data_format": data_format,
                    "resolution": resolution,
                },
            ],
            uniform_yaw_distribution=uniform_yaw_distribution,
            generators_count=generators_count,
        )

    def __len__(self):
        return self.max_samples

    def __getitem__(self, idx):
        warped, target, target_mask, target_part_mask = self.generator.generate_next()[0]

        warped = torch.from_numpy(warped).float()
        target = torch.from_numpy(target).float()
        target_mask = torch.from_numpy(target_mask).float()
        target_part_mask = torch.from_numpy(target_part_mask).float()

        return warped[0], target[0], target_mask[0], target_part_mask[0]
