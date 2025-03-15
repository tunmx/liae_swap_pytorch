import os
import sys
import click
import cv2
import importlib
import numpy as np
import torch
from adabelief_pytorch import AdaBelief
from torch.cuda.amp import GradScaler, autocast
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from data import FaceSwapDataset
from loss import Criterion
from model import LIAE
from utils.image_process import GaussianBlur, BackgroundBlur, draw_visualization_predicted, draw_loss_plot, draw_fixed_predicted
from utils.logger import SimpleLogger
from utils.scheduler import WarmupCosineScheduler, init_seed
from utils.trainer import Trainer

class DeepFakeTrainer(Trainer):
    
    def __init__(self, config):
        self.cfg = config
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        torch.cuda.set_device(self.device)
        
        os.makedirs(self.cfg.exp_name, exist_ok=True)
        os.makedirs(self.cfg.save_model_path, exist_ok=True)
        os.makedirs(self.cfg.results_grid_path, exist_ok=True)
        
        self.logger = SimpleLogger("Logger", filename=os.path.join(self.cfg.exp_name, "training.log"))
        
        init_seed()
        
        self.epoch = 0
        self.training_step = 0
        self.loss_check_threshold = 10000
        self.mean_loss_src = []
        self.mean_loss_dst = []
        self.visualize_once_flag = self.cfg.visualize_once_flag

        self.best_src_loss = float('inf')
        self.best_dst_loss = float('inf')
        self.best_src_iter = 0
        self.best_dst_iter = 0
        
        grid_num = min(self.cfg.show_grid_num, self.cfg.batch_size)
        if grid_num % 2 != 0:  
            grid_num = min(grid_num + 1, self.cfg.batch_size)  
            if grid_num > self.cfg.batch_size:  
                grid_num -= 2  
            self.logger.warn(f"Adjusted show_grid_num to {grid_num}")
        self.show_grid_num = grid_num
        
    def setup_models(self):
        self.model = LIAE(
            opts=self.cfg.opts,
            resolution=self.cfg.resolution,
            input_channels=self.cfg.input_channels,
            e_dims=self.cfg.e_dims,
            ae_dims=self.cfg.ae_dims,
            d_dims=self.cfg.d_dims,
            d_mask_dims=self.cfg.d_mask_dims,
        )
        
        self.encoder = self.model.encoder.to(self.device)
        self.inter_AB = self.model.inter_AB.to(self.device)
        self.inter_B = self.model.inter_B.to(self.device)
        self.decoder = self.model.decoder.to(self.device)
    

    def setup_dataloaders(self):
        self.dataset_src = FaceSwapDataset(
            data_path=self.cfg.src_aligned_path,
            dataset_type='src',
            random_ct_samples_path=self.cfg.random_ct_samples_path,
            is_debug=self.cfg.is_debug,
            random_flip=self.cfg.random_src_flip,
            random_warp=self.cfg.random_warp,
            ct_mode=self.cfg.ct_mode,
            random_hsv_power=self.cfg.random_hsv_power,
            face_type=self.cfg.face_type,
            data_format=self.cfg.data_format,
            resolution=self.cfg.resolution,
            uniform_yaw_distribution=self.cfg.uniform_yaw,
            generators_count=self.cfg.src_generators_count,
            pak_name=self.cfg.src_pak_name,
            ignore_same_path=self.cfg.ignore_same_path,
            scale_range=self.cfg.scale_range,
            random_downsample=self.cfg.random_downsample,
            random_noise=self.cfg.random_noise,
            random_blur=self.cfg.random_blur,
            random_jpeg=self.cfg.random_jpeg,
            random_shadow=self.cfg.random_shadow,
        )
        
        self.dataset_dst = FaceSwapDataset(
            data_path=self.cfg.dst_aligned_path,
            dataset_type='dst',
            random_ct_samples_path=self.cfg.random_ct_samples_path,
            is_debug=self.cfg.is_debug,
            random_flip=self.cfg.random_dst_flip,
            random_warp=self.cfg.random_warp,
            ct_mode=self.cfg.ct_mode,
            random_hsv_power=self.cfg.random_hsv_power,
            face_type=self.cfg.face_type,
            data_format=self.cfg.data_format,
            resolution=self.cfg.resolution,
            uniform_yaw_distribution=self.cfg.uniform_yaw,
            generators_count=self.cfg.dst_generators_count,
            pak_name=self.cfg.dst_pak_name,
            ignore_same_path=self.cfg.ignore_same_path,
            scale_range=self.cfg.scale_range,
            random_downsample=self.cfg.random_downsample,
            random_noise=self.cfg.random_noise,
            random_blur=self.cfg.random_blur,
            random_jpeg=self.cfg.random_jpeg,
            random_shadow=self.cfg.random_shadow,
        )
        
        self.data_loader_src = DataLoader(
            self.dataset_src,
            batch_size=self.cfg.batch_size,
            shuffle=True,
            num_workers=self.cfg.num_workers,
            persistent_workers=True,
            pin_memory=True,
            drop_last=True,
        )
        
        self.data_loader_dst = DataLoader(
            self.dataset_dst,
            batch_size=self.cfg.batch_size,
            shuffle=True,
            num_workers=self.cfg.num_workers,
            persistent_workers=True,
            pin_memory=True,
            drop_last=True,
        )
    

    def setup_training(self):
        self.optimizer = AdaBelief(
            [
                {
                    "params": list(self.encoder.parameters())
                    + list(self.inter_AB.parameters())
                    + list(self.inter_B.parameters())
                    + list(self.decoder.parameters())
                },
            ],
            lr=self.cfg.lr,
            betas=(0.5, 0.999),
            print_change_log=False,
        )
        
        self.scheduler = WarmupCosineScheduler(
            self.optimizer, 
            self.cfg.lr, 
            self.cfg.total_iter, 
            self.cfg.warmup_iter, 
            self.cfg.lr_dropout, 
            self.cfg.lr_cos
        )
        
        self.criterion = Criterion(
            self.cfg.resolution, 
            self.cfg.use_part_loss, 
            self.device,
            self.cfg.ssim_weight,
            self.cfg.seg_mse_weight,
            self.cfg.mask_mse_weight,
            self.cfg.part_prio_weight
        )
        
        self.scaler = GradScaler(enabled=self.cfg.fp16)
        
        self.mask_blur = GaussianBlur(
            radius=self.cfg.resolution // 32, 
            channels=1, 
            device=self.device
        )
        
        self.background_blur = BackgroundBlur(
            radius=self.cfg.resolution // 128, 
            channels=3, 
            device=self.device
        )


    def load_checkpoint(self):
        if self.cfg.pretrained_model_path is not None:
            ckpts_path = f"{self.cfg.pretrained_model_path}"
            if os.path.exists(ckpts_path):
                ckpts = torch.load(ckpts_path)
                
                self.encoder.load_state_dict(ckpts["encoder"])
                self.inter_B.load_state_dict(ckpts["inter_B"])
                self.inter_AB.load_state_dict(ckpts["inter_AB"])
                self.decoder.load_state_dict(ckpts["decoder"])
                
                self.optimizer.load_state_dict(ckpts["optimizer"])
                if ckpts["scaler"]:
                    self.scaler.load_state_dict(ckpts["scaler"])
                    
                self.training_step = ckpts["global_step"]

                # Load best loss values if available, otherwise use defaults
                self.best_src_loss = ckpts.get("best_src_loss", float('inf'))
                self.best_dst_loss = ckpts.get("best_dst_loss", float('inf'))
                self.best_src_iter = ckpts.get("best_src_iter", 0)
                self.best_dst_iter = ckpts.get("best_dst_iter", 0)
                
                self.logger.info(f"Resume from {self.cfg.pretrained_model_path}")
                self.logger.info(f"Best SRC loss: {self.best_src_loss:.5f} at epoch {self.best_src_iter}")
                self.logger.info(f"Best DST loss: {self.best_dst_loss:.5f} at epoch {self.best_dst_iter}")
                    
                self.cfg.total_iter += self.training_step
            else:
                self.logger.err(f"{ckpts_path} not found!")
                sys.exit(1)

    def save_checkpoint(self):
        checkpoint = {
            "encoder": self.encoder.state_dict(),
            "inter_AB": self.inter_AB.state_dict(),
            "inter_B": self.inter_B.state_dict(),
            "decoder": self.decoder.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scaler": self.scaler.state_dict(),
            "global_step": self.training_step,
            "best_src_loss": self.best_src_loss,
            "best_dst_loss": self.best_dst_loss,
            "best_src_iter": self.best_src_iter,
            "best_dst_iter": self.best_dst_iter,
        }
        
        checkpoint_path = f"{self.cfg.save_model_path}/ckpts_{self.training_step}.pt"
        torch.save(checkpoint, checkpoint_path)
        self.logger.info(f"Save model {checkpoint_path}")
        
    def prepare_debug_directory(self):
        if self.visualize_once_flag:
            self.data_processed_path = os.path.join(self.cfg.exp_name, "data_processed")
            os.makedirs(self.data_processed_path, exist_ok=True)

    def save_debug_image(self, image, name):
        image_np = image.cpu().numpy()
        image_np = image_np.transpose(1, 2, 0)
        image_np = image_np * 255
        image_np = image_np.astype(np.uint8)
        
        cv2.imwrite(os.path.join(self.data_processed_path, f"{name}.png"), image_np)

    def process_masks(self, target_src, target_dst, target_src_mask, target_dst_mask, target_src_part_mask, target_dst_part_mask):
        target_src_mask = torch.clamp(target_src_mask, 0, 1)
        target_dst_mask = torch.clamp(target_dst_mask, 0, 1)
        
        if self.cfg.use_part_loss:
            target_src_part_mask = torch.clamp(target_src_part_mask - 1, 0, 1).to(self.device)
            target_dst_part_mask = torch.clamp(target_dst_part_mask - 1, 0, 1).to(self.device)
        else:
            target_src_part_mask = None
            target_dst_part_mask = None
        
        target_src_mask_blur = self.mask_blur(target_src_mask)
        target_src_mask_blur = torch.clamp(target_src_mask_blur, 0, 0.5) * 2
        
        target_dst_mask_blur = self.mask_blur(target_dst_mask)
        target_dst_mask_blur = torch.clamp(target_dst_mask_blur, 0, 0.5) * 2
        
        target_src_segmented = target_src * target_src_mask_blur if self.cfg.masked_training else target_src
        target_dst_segmented = target_dst * target_dst_mask_blur if self.cfg.masked_training else target_dst
        
        return (
            target_src_mask,
            target_dst_mask,
            target_src_part_mask,
            target_dst_part_mask,
            target_src_mask_blur,
            target_dst_mask_blur,
            target_src_segmented,
            target_dst_segmented
        )
        
    def train_source_image(self, warped_src, target_src, target_src_mask, target_src_part_mask, target_src_segmented, target_src_mask_blur):
        with autocast(enabled=self.cfg.fp16, dtype=torch.float16):
            src_code = self.encoder(warped_src)
            
            src_inter_AB_code = self.inter_AB(src_code)
            
            src_code = torch.cat([src_inter_AB_code, src_inter_AB_code], dim=1)
            
            pred_src, pred_src_mask = self.decoder(src_code)
            
            pred_src_masked = pred_src * target_src_mask_blur if self.cfg.masked_training else pred_src
            
            losses = self.criterion(
                pred_src_masked,
                target_src_segmented,
                pred_src_mask,
                target_src_mask,
                pred_src,
                target_src,
                target_src_part_mask,
            )
            
            loss = sum(losses[k] for k in losses) / self.cfg.grad_accum_steps
            loss_all = loss.item() * self.cfg.grad_accum_steps
            
            if np.isnan(loss_all) or loss_all > 3 * self.loss_check_threshold:
                self.logger.warn(f"Abnormal loss value: {loss_all}")
            self.loss_check_threshold = loss_all
        
        self.scaler.scale(loss).backward()
        
        if self.training_step % self.cfg.log_interval == 0:
            if loss_all < self.best_src_loss:
                self.best_src_loss = loss_all
                self.best_src_iter = self.training_step

            loss_str = " ".join([f"{k}:{v.item():.5f}" for k, v in losses.items()])
            self.logger.info(f"[Iter {self.training_step:06d}] │ SRC │ {loss_str} │ Total: {loss_all:.5f} │ Best: {self.best_src_loss:.5f} at iter {self.best_src_iter}")
            self.mean_loss_src.append(loss_all)
        
        return pred_src
    
    def train_target_image(self, warped_dst, target_dst, target_dst_mask, target_dst_part_mask, target_dst_segmented, target_dst_mask_blur):
        with autocast(enabled=self.cfg.fp16, dtype=torch.float16):
            dst_code = self.encoder(warped_dst)
            
            dst_inter_B_code = self.inter_B(dst_code)
            dst_inter_AB_code = self.inter_AB(dst_code)
            
            dst_code = torch.cat([dst_inter_B_code, dst_inter_AB_code], dim=1)
            
            pred_dst, pred_dst_mask = self.decoder(dst_code)
            
            pred_dst_masked = pred_dst * target_dst_mask_blur if self.cfg.masked_training else pred_dst
            
            losses = self.criterion(
                pred_dst_masked,
                target_dst_segmented,
                pred_dst_mask,
                target_dst_mask,
                pred_dst,
                target_dst,
                target_dst_part_mask,
            )
            
            loss = sum(losses[k] for k in losses) / self.cfg.grad_accum_steps
            loss_all = loss.item() * self.cfg.grad_accum_steps
            
            with torch.no_grad():
                src_dst_code = torch.cat([dst_inter_AB_code, dst_inter_AB_code], dim=1)
                fake, _ = self.decoder(src_dst_code)
        
        self.scaler.scale(loss).backward()
        
        if self.cfg.clip_grad > 0:
            self.scaler.unscale_(self.optimizer)
            clip_grad_norm_(self.encoder.parameters(), self.cfg.clip_grad)
            clip_grad_norm_(self.inter_B.parameters(), self.cfg.clip_grad)
            clip_grad_norm_(self.inter_AB.parameters(), self.cfg.clip_grad)
            clip_grad_norm_(self.decoder.parameters(), self.cfg.clip_grad)
        

        if self.training_step % self.cfg.log_interval == 0:
            if loss_all < self.best_dst_loss:
                self.best_dst_loss = loss_all
                self.best_dst_iter = self.training_step

            loss_str = " ".join([f"{k}:{v.item():.5f}" for k, v in losses.items()])
            self.logger.info(f"[Iter {self.training_step:06d}] │ DST │ {loss_str} │ Total: {loss_all:.5f} │ Best: {self.best_dst_loss:.5f} at iter {self.best_dst_iter}")
            self.mean_loss_dst.append(loss_all)
        
        return pred_dst, fake
    
    def save_visualization_debug(self, data_dict):
        if not self.visualize_once_flag:
            return
        
        for name, image in data_dict.items():
            self.save_debug_image(image, name)
        
        self.logger.info(f"target_src shape: {data_dict['target_src'].shape}")
        self.logger.info(f"Save visualize results to {self.data_processed_path}")
        self.visualize_once_flag = False

    def train_step(self, data_src, data_dst):
        warped_src, target_src, target_src_mask, target_src_part_mask = data_src
        warped_dst, target_dst, target_dst_mask, target_dst_part_mask = data_dst
        
        warped_src = warped_src.to(self.device)
        target_src = target_src.to(self.device)
        target_src_mask = target_src_mask.to(self.device)
        
        warped_dst = warped_dst.to(self.device)
        target_dst = target_dst.to(self.device)
        target_dst_mask = target_dst_mask.to(self.device)
        
        target_src = self.background_blur(target_src, target_src_mask)
        target_dst = self.background_blur(target_dst, target_dst_mask)
        
        (
            target_src_mask,
            target_dst_mask,
            target_src_part_mask,
            target_dst_part_mask,
            target_src_mask_blur,
            target_dst_mask_blur,
            target_src_segmented,
            target_dst_segmented
        ) = self.process_masks(
            target_src,  
            target_dst,  
            target_src_mask, 
            target_dst_mask, 
            target_src_part_mask, 
            target_dst_part_mask
        )
        
        if self.visualize_once_flag:
            debug_images = {
                'target_src_blur_before': target_src[0],
                'target_dst_blur_before': target_dst[0],
                'target_src_segmented': target_src_segmented[0],
                'target_dst_segmented': target_dst_segmented[0],
                'target_src_mask_blur': target_src_mask_blur[0],
                'target_dst_mask_blur': target_dst_mask_blur[0],
                'warped_src': warped_src[0],
                'warped_dst': warped_dst[0],
                'target_src': target_src[0],
                'target_dst': target_dst[0],
                'target_src_mask': target_src_mask[0],
                'target_dst_mask': target_dst_mask[0],
            }
            
            if self.cfg.use_part_loss:
                debug_images['target_src_part_mask'] = target_src_part_mask[0]
                debug_images['target_dst_part_mask'] = target_dst_part_mask[0]
                
            self.save_visualization_debug(debug_images)
        
        self.optimizer.zero_grad()
        
        pred_src = self.train_source_image(
            warped_src, 
            target_src, 
            target_src_mask, 
            target_src_part_mask, 
            target_src_segmented, 
            target_src_mask_blur
        )
        
        pred_dst, fake = self.train_target_image(
            warped_dst, 
            target_dst, 
            target_dst_mask, 
            target_dst_part_mask, 
            target_dst_segmented, 
            target_dst_mask_blur
        )
        
        if self.training_step % self.cfg.grad_accum_steps == 0:
            self.scaler.step(self.optimizer)
            self.scaler.update()
        
        if self.cfg.lr_dropout:
            cur_lr = self.scheduler.step(self.training_step)
        else:
            cur_lr = self.cfg.lr
        
        if self.training_step > 0 and self.training_step % self.cfg.save_model_interval == 0:
            self.save_checkpoint()
            self.logger.info(f"Current LR: {cur_lr:.2e}")
        
        if self.training_step > 0 and self.training_step % self.cfg.draw_results_interval == 0:
            draw_loss_plot(self.mean_loss_src, self.mean_loss_dst, self.training_step, f"{self.cfg.exp_name}/loss.jpg")

            grid_result = draw_visualization_predicted(
                pred_src,
                target_src,
                pred_dst,
                target_dst,
                fake,
                self.show_grid_num,
            )
            cv2.imwrite(f"{self.cfg.results_grid_path}/{self.training_step}.jpg", grid_result)
            if self.cfg.test_fixed_samples_path != None:
                self.eval_fixed_samples()
    

    def setup_fixed_samples(self):
        if self.cfg.test_fixed_samples_path is None:
            self.logger.info("If the fixed sample path is not set, the fixed sample evaluation will not be used.")
            self.use_fixed_samples = False
            return
        
        self.use_fixed_samples = True
        self.fixed_samples_path = self.cfg.test_fixed_samples_path
        self.fixed_batch_size = self.cfg.test_fixed_samples_batch_size
        
        os.makedirs(self.fixed_samples_path, exist_ok=True)
        
        src_samples = [f for f in os.listdir(self.fixed_samples_path) if f.startswith('sample_src_') and f.endswith('.npy')]
        dst_samples = [f for f in os.listdir(self.fixed_samples_path) if f.startswith('sample_dst_') and f.endswith('.npy')]
        
        valid_samples = len(src_samples) == len(dst_samples) and len(src_samples) >= self.fixed_batch_size
        
        if not valid_samples:
            self.logger.info("The fixed samples do not exist or the number does not match, the new fixed samples will be generated from the data loader.")
            
            if not hasattr(self, 'data_loader_src') or not hasattr(self, 'data_loader_dst'):
                self.logger.info("Please set the data loader first.")
                self.setup_dataloaders()
            
            for f in os.listdir(self.fixed_samples_path):
                if f.startswith('sample_src_') or f.startswith('sample_dst_'):
                    os.remove(os.path.join(self.fixed_samples_path, f))
            
            try:
                data_src = next(iter(self.data_loader_src))
                data_dst = next(iter(self.data_loader_dst))
                
                warped_src, target_src, target_src_mask, target_src_part_mask = data_src
                warped_dst, target_dst, target_dst_mask, target_dst_part_mask = data_dst
                
                num_samples = min(self.fixed_batch_size, len(warped_src))
                
                for i in range(num_samples):
                    src_sample = (
                        warped_src[i].cpu().numpy(),
                        target_src[i].cpu().numpy(),
                        target_src_mask[i].cpu().numpy(),
                        target_src_part_mask[i].cpu().numpy() if target_src_part_mask is not None else None
                    )
                    np.save(os.path.join(self.fixed_samples_path, f'sample_src_{i}.npy'), src_sample)
                    
                    dst_sample = (
                        warped_dst[i].cpu().numpy(),
                        target_dst[i].cpu().numpy(),
                        target_dst_mask[i].cpu().numpy(),
                        target_dst_part_mask[i].cpu().numpy() if target_dst_part_mask is not None else None
                    )
                    np.save(os.path.join(self.fixed_samples_path, f'sample_dst_{i}.npy'), dst_sample)
                
                self.logger.info(f"Created {num_samples} pairs of fixed samples.")
                self.fixed_samples_created = True
                
            except Exception as e:
                self.logger.warn(f"Failed to create fixed samples: {str(e)}")
                self.use_fixed_samples = False
                return
        else:
            self.logger.info(f"Found {len(src_samples)} pairs of fixed samples.")
            self.fixed_samples_created = True
        
        try:
            src_samples = sorted([f for f in os.listdir(self.fixed_samples_path) if f.startswith('sample_src_') and f.endswith('.npy')])
            dst_samples = sorted([f for f in os.listdir(self.fixed_samples_path) if f.startswith('sample_dst_') and f.endswith('.npy')])
            
            src_samples = src_samples[:self.fixed_batch_size]
            dst_samples = dst_samples[:self.fixed_batch_size]
            
            self.fixed_sample_src = []
            self.fixed_sample_dst = []
            
            for src_file, dst_file in zip(src_samples, dst_samples):
                src_path = os.path.join(self.fixed_samples_path, src_file)
                src_data = np.load(src_path, allow_pickle=True)
                
                warped_src = torch.from_numpy(src_data[0]).float().to(self.device)
                target_src = torch.from_numpy(src_data[1]).float().to(self.device)
                target_src_mask = torch.from_numpy(src_data[2]).float().to(self.device)
                
                if src_data[3] is not None:
                    target_src_part_mask = torch.from_numpy(src_data[3]).float().to(self.device)
                else:
                    target_src_part_mask = None
                
                self.fixed_sample_src.append((warped_src, target_src, target_src_mask, target_src_part_mask))
                
                dst_path = os.path.join(self.fixed_samples_path, dst_file)
                dst_data = np.load(dst_path, allow_pickle=True)
                
                warped_dst = torch.from_numpy(dst_data[0]).float().to(self.device)
                target_dst = torch.from_numpy(dst_data[1]).float().to(self.device)
                target_dst_mask = torch.from_numpy(dst_data[2]).float().to(self.device)
                
                if dst_data[3] is not None:
                    target_dst_part_mask = torch.from_numpy(dst_data[3]).float().to(self.device)
                else:
                    target_dst_part_mask = None
                
                self.fixed_sample_dst.append((warped_dst, target_dst, target_dst_mask, target_dst_part_mask))
            
            self.logger.info(f"Loaded {len(self.fixed_sample_src)} pairs of fixed samples to the device.")
            
        except Exception as e:
            self.logger.warn(f"Failed to load fixed samples to the device: {str(e)}")
            self.use_fixed_samples = False


    def eval_fixed_samples(self):
        if not self.use_fixed_samples or not hasattr(self, 'fixed_sample_src') or len(self.fixed_sample_src) == 0:
            self.logger.info("No available fixed samples, skip evaluation.")
            return
        
        self.test_fixed_results = os.path.join(self.cfg.exp_name, "test_fixed_results")
        os.makedirs(self.test_fixed_results, exist_ok=True)
            
        self.encoder.eval()
        self.inter_AB.eval()
        self.inter_B.eval()
        self.decoder.eval()
        
        with torch.no_grad():
            batch_size = len(self.fixed_sample_src)
            warped_src_list = []
            target_src_list = []
            target_src_mask_list = []
            target_src_part_mask_list = []
            
            warped_dst_list = []
            target_dst_list = []
            target_dst_mask_list = []
            target_dst_part_mask_list = []
            
            for i in range(batch_size):
                warped_src, target_src, target_src_mask, target_src_part_mask = self.fixed_sample_src[i]
                warped_dst, target_dst, target_dst_mask, target_dst_part_mask = self.fixed_sample_dst[i]
                
                warped_src_list.append(warped_src)
                target_src_list.append(target_src)
                target_src_mask_list.append(target_src_mask)
                if target_src_part_mask is not None:
                    target_src_part_mask_list.append(target_src_part_mask)
                
                warped_dst_list.append(warped_dst)
                target_dst_list.append(target_dst)
                target_dst_mask_list.append(target_dst_mask)
                if target_dst_part_mask is not None:
                    target_dst_part_mask_list.append(target_dst_part_mask)
            
            warped_src_batch = torch.stack(warped_src_list)
            target_src_batch = torch.stack(target_src_list)
            target_src_mask_batch = torch.stack(target_src_mask_list)
            
            warped_dst_batch = torch.stack(warped_dst_list)
            target_dst_batch = torch.stack(target_dst_list)
            target_dst_mask_batch = torch.stack(target_dst_mask_list)
            
            if len(target_src_part_mask_list) > 0:
                target_src_part_mask_batch = torch.stack(target_src_part_mask_list)
            else:
                target_src_part_mask_batch = None
                
            if len(target_dst_part_mask_list) > 0:
                target_dst_part_mask_batch = torch.stack(target_dst_part_mask_list)
            else:
                target_dst_part_mask_batch = None
            
            target_src_batch = self.background_blur(target_src_batch, target_src_mask_batch)
            target_dst_batch = self.background_blur(target_dst_batch, target_dst_mask_batch)
            
            (
                target_src_mask_batch,
                target_dst_mask_batch,
                target_src_part_mask_batch,
                target_dst_part_mask_batch,
                target_src_mask_blur_batch,
                target_dst_mask_blur_batch,
                target_src_segmented_batch,
                target_dst_segmented_batch
            ) = self.process_masks(
                target_src_batch,  
                target_dst_batch,  
                target_src_mask_batch, 
                target_dst_mask_batch, 
                target_src_part_mask_batch, 
                target_dst_part_mask_batch
            )
            
            src_code = self.encoder(warped_src_batch)
            src_inter_AB_code = self.inter_AB(src_code)
            src_code = torch.cat([src_inter_AB_code, src_inter_AB_code], dim=1)
            pred_src, _ = self.decoder(src_code)
            
            dst_code = self.encoder(warped_dst_batch)
            dst_inter_B_code = self.inter_B(dst_code)
            dst_inter_AB_code = self.inter_AB(dst_code)
            dst_code = torch.cat([dst_inter_B_code, dst_inter_AB_code], dim=1)
            pred_dst, _ = self.decoder(dst_code)
            
            src_dst_code = torch.cat([dst_inter_AB_code, dst_inter_AB_code], dim=1)
            fake, _ = self.decoder(src_dst_code)
            
            grid_result = draw_fixed_predicted(
                pred_src,
                pred_dst,
                fake,
            )
            
            cv2.imwrite(f"{self.test_fixed_results}/fixed_{self.training_step}.jpg", grid_result)
            
        self.encoder.train()
        self.inter_AB.train()
        self.inter_B.train()
        self.decoder.train()
        

    def fit(self):
        self.setup_models()
        self.setup_dataloaders()
        self.setup_fixed_samples()
        self.setup_training()
        self.load_checkpoint()
        self.prepare_debug_directory()
        
        while self.training_step < self.cfg.total_iter:
            self.epoch += 1
            for i, (data_src, data_dst) in enumerate(zip(self.data_loader_src, self.data_loader_dst)):
                self.training_step += 1
                
                self.train_step(data_src, data_dst)
                
                if self.training_step >= self.cfg.total_iter:
                    break

    
@click.command()
@click.option("--config", type=str, default="configs/config.py", help="Config file path")
def prepare(**kwargs):
    config_path = kwargs['config']
    spec = importlib.util.spec_from_file_location("configs", config_path)
    config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config)
    
    exp_name = os.path.join("workspace", config.exp_name)
    config.exp_name = exp_name
    config.save_model_path = os.path.join(exp_name, "ckpts")
    config.results_grid_path = os.path.join(exp_name, "grid_results")
    
    logtxt = os.path.join(config.exp_name, "config.txt")
    os.makedirs(config.exp_name, exist_ok=True)
    with open(logtxt, "w") as flog:
        for name, value in vars(config).items():
            if name != "sys" and not name.startswith("__") and not callable(value):
                flog.write(f"{name}: {value}\n")
    
    trainer = DeepFakeTrainer(config)
    
    trainer.fit()

if __name__ == "__main__":
    prepare()