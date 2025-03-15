
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2

class DSSIMLoss(nn.Module):

    def __init__(self, filter_size=11, device=None, max_val=1, k1=0.01, k2=0.03):
        super(DSSIMLoss, self).__init__()
        self.max_val = max_val
        self.filter_size = filter_size
        self.device = device
        self.filter_sigma = 1.5
        self.k1 = k1
        self.k2 = k2
        
        self.c1 = (self.k1 * self.max_val) ** 2
        self.c2 = (self.k2 * self.max_val) ** 2
        
        self._create_gaussian_window()
    
    def _create_gaussian_window(self):
        gaussian_1d = cv2.getGaussianKernel(self.filter_size, self.filter_sigma)
        gaussian_1d = torch.tensor(gaussian_1d, dtype=torch.float32, device=self.device)
        
        gaussian_2d = gaussian_1d @ gaussian_1d.t()
        
        self.window = gaussian_2d.expand(3, 1, self.filter_size, self.filter_size).contiguous()
    
    def _compute_local_stats(self, x, y):
        mu_x = F.conv2d(x, self.window, padding="valid", groups=3)
        mu_y = F.conv2d(y, self.window, padding="valid", groups=3)
        
        mu_x_sq = mu_x.pow(2)
        mu_y_sq = mu_y.pow(2)
        mu_xy = mu_x * mu_y
        
        var_x = F.conv2d(x * x, self.window, padding="valid", groups=3) - mu_x_sq + 1e-6
        var_y = F.conv2d(y * y, self.window, padding="valid", groups=3) - mu_y_sq + 1e-6
        cov_xy = F.conv2d(x * y, self.window, padding="valid", groups=3) - mu_xy + 1e-6
        
        return mu_x, mu_y, mu_x_sq, mu_y_sq, var_x, var_y, cov_xy
    
    def forward(self, img1, img2):
        mu1, mu2, mu1_sq, mu2_sq, sigma1_sq, sigma2_sq, sigma12 = self._compute_local_stats(img1, img2)
        
        luminance_term = (2.0 * mu1 * mu2 + self.c1) / (mu1_sq + mu2_sq + self.c1)
        contrast_term = (2.0 * sigma12 + self.c2) / (sigma1_sq + sigma2_sq + self.c2)
        
        ssim_map = luminance_term * contrast_term
        ssim_map = torch.clamp(ssim_map, min=0.0)  
        
        return (1.0 - ssim_map.mean()) / 2.0

class Criterion(nn.Module):

    def __init__(
        self, 
        resolution, 
        part_prio=False, 
        device=None,
        ssim_weight=5,
        seg_mse_weight=10,
        mask_mse_weight=10,
        part_prio_weight=300
    ):
        super(Criterion, self).__init__()
        self.resolution = resolution
        self.part_prio = part_prio
        self.ssim_weight = ssim_weight
        self.seg_mse_weight = seg_mse_weight
        self.mask_mse_weight = mask_mse_weight
        self.part_prio_weight = part_prio_weight
        
        k1 = 0.01
        if self.resolution > 512:
            k2 = 0.4
        else:
            k2 = 0.03

        self.ssim_loss1 = DSSIMLoss(
            filter_size=int(self.resolution / 11.6), device=device, k1=k1, k2=k2
        )
        self.ssim_loss2 = DSSIMLoss(
            filter_size=int(self.resolution / 23.2), device=device, k1=k1, k2=k2
        )
        
        self.seg_mse_loss = nn.MSELoss()
        self.mask_mse_loss = nn.MSELoss()

        if self.part_prio:
            self.part_prio_loss = nn.L1Loss()

    def forward(
        self, 
        pred_segmented,         
        target_segmented,       
        pred_mask,              
        target_mask,            
        pred_face,              
        target_face,            
        target_part_mask, 
    ):
        """
        Compute the combined loss for image segmentation.
        
        For high resolution images (>256), uses two SSIM losses with different scales.
        For lower resolutions, uses a single SSIM loss with double weight.
        Always includes MSE losses for segmentation and mask.
        Optionally includes priority loss for facial parts if enabled.
        """
        if self.resolution > 256:
            loss_ssim = self.ssim_weight * self.ssim_loss1(pred_segmented, target_segmented) \
                + self.ssim_weight * self.ssim_loss2(pred_segmented, target_segmented)
        else:
            loss_ssim = 2 * self.ssim_weight * self.ssim_loss1(pred_segmented, target_segmented)
        
        loss_seg_mse = self.seg_mse_weight * self.seg_mse_loss(pred_segmented, target_segmented)
        loss_mask_mse = self.mask_mse_weight * self.mask_mse_loss(pred_mask, target_mask)
        
        loss = dict(loss_ssim=loss_ssim, loss_seg_mse=loss_seg_mse, loss_mask_mse=loss_mask_mse)

        if self.part_prio:
            loss_part_prio = self.part_prio_weight * self.part_prio_loss(
                pred_face * target_part_mask, 
                target_face * target_part_mask
            )
            loss["loss_part_prio"] = loss_part_prio
        
        return loss