import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import torchvision

class GaussianBlur:
    def __init__(self, radius=2.0, channels=1, device=torch.device("cpu")):
        self.kernel, self.kernel_size = self.make_kernel(radius)
        self.padding = self.kernel_size // 2
        self.kernel = self.kernel.view(1, 1, self.kernel_size, self.kernel_size)
        self.kernel = self.kernel.repeat(channels, 1, 1, 1).to(device)
        self.kernel.requires_grad_(False)

    @staticmethod
    def gaussian(x, mu, sigma):
        return np.exp(-((float(x) - float(mu)) ** 2) / (2 * sigma**2))

    @staticmethod
    def make_kernel(sigma):
        kernel_size = max(3, int(2 * 2 * sigma))
        if kernel_size % 2 == 0:
            kernel_size += 1
        mean = (kernel_size - 1) / 2.0
        grid = torch.arange(kernel_size, dtype=torch.float32)
        grid = grid - mean
        kernel_1d = torch.exp(-0.5 * (grid / sigma) ** 2)
        kernel_1d = kernel_1d / kernel_1d.sum()
        kernel_2d = torch.outer(kernel_1d, kernel_1d)
        return kernel_2d, kernel_size

    def __call__(self, img):
        if self.padding != 0:
            input_padded = F.pad(
                img,
                (self.padding, self.padding, self.padding, self.padding),
                mode="constant",
            )
        else:
            input_padded = img
        blurred = F.conv2d(input_padded, self.kernel, groups=img.shape[1], padding=0)
        return blurred

class BackgroundBlur:
    def __init__(self, radius=2.0, channels=1, device=torch.device("cpu")):
        self.blur = GaussianBlur(radius, channels, device)

    def __call__(self, img, salient_mask):
        anti_mask = 1 - salient_mask
        
        masked_img = img * anti_mask
        blurred_bg = self.blur(masked_img)
        
        blurred_mask = self.blur(salient_mask)
        inverted_blurred_mask = 1 - blurred_mask
        
        safe_divisor = torch.clone(inverted_blurred_mask)
        safe_divisor[safe_divisor == 0] = 1.0 
        
        foreground = img * salient_mask
        normalized_bg = blurred_bg / safe_divisor
        background = normalized_bg * anti_mask
        result = foreground + background

        return result

class BrightnessAdjust:
    def __init__(self, brightness_factor=1.0, device=torch.device("cpu")):
        self.brightness_factor = brightness_factor
        self.device = device
        self.identity = torch.eye(3, device=device)
        self.brightness_matrix = self._make_brightness_matrix(brightness_factor)
        self.brightness_matrix = self.brightness_matrix.to(device)
        self.brightness_matrix.requires_grad_(False)
    
    def _make_brightness_matrix(self, factor):
        matrix = self.identity.clone()
        matrix = matrix * factor
        return matrix
    
    def set_brightness(self, new_factor):
        self.brightness_factor = new_factor
        self.brightness_factor = new_factor
        self.brightness_matrix = self._make_brightness_matrix(new_factor)
        self.brightness_matrix = self.brightness_matrix.to(self.device)
        
    def __call__(self, img):
        adjusted = img * self.brightness_factor
        adjusted = torch.clamp(adjusted, 0.0, 1.0)
        
        return adjusted
    
    def apply_with_mask(self, img, mask):
        adjusted = img * self.brightness_factor
        
        adjusted = torch.clamp(adjusted, 0.0, 1.0)
        
        result = mask * adjusted + (1 - mask) * img
        
        return result

class NoiseAdder:
    def __init__(self, noise_level=0.1, noise_type='gaussian', device=torch.device("cpu")):
        self.noise_level = noise_level
        self.noise_type = noise_type
        self.device = device
    
    def set_noise_level(self, new_level):
        self.noise_level = new_level
    
    def set_noise_type(self, new_type):
        if new_type not in ['gaussian', 'salt_pepper', 'poisson']:
            raise ValueError("Noise type must be 'gaussian', 'salt_pepper', 'poisson'")
        self.noise_type = new_type
        
    def __call__(self, img):
        if self.noise_type == 'gaussian':
            noise = torch.randn_like(img, device=self.device) * self.noise_level
            noisy_img = img + noise
            return torch.clamp(noisy_img, 0.0, 1.0)
        
        elif self.noise_type == 'salt_pepper':
            noisy_img = img.clone()
            batch_size, channels, height, width = img.shape
            
            num_salt = int(self.noise_level * img.numel() * 0.5)
            num_pepper = int(self.noise_level * img.numel() * 0.5)
            
            salt_indices = torch.randint(0, img.numel(), (num_salt,), device=self.device)
            pepper_indices = torch.randint(0, img.numel(), (num_pepper,), device=self.device)
            
            flat_img = noisy_img.view(-1)
            flat_img[salt_indices] = 1.0
            flat_img[pepper_indices] = 0.0
            
            return noisy_img
            
        elif self.noise_type == 'poisson':
            scaling = 10.0
            noisy_img = torch.poisson(img * scaling) / scaling
            return torch.clamp(noisy_img, 0.0, 1.0)
            
    def apply_with_mask(self, img, mask):
        noisy_img = self(img)
        result = mask * noisy_img + (1 - mask) * img
        return result

def draw_loss_plot(loss_src, loss_dst, steps, save_path):
    plt.figure(figsize=(10, 6))
    plt.plot(loss_src, label="loss src")
    plt.plot(loss_dst, label="loss dst")
    
    plt.legend()
    plt.title(f"Steps: {steps}")
    
    plt.savefig(save_path)
    plt.close()

def draw_fixed_predicted(reconstruct_src, reconstruct_dst, fake):
    images_for_grid = []
    for ii in range(4):
        images_for_grid.extend(
            [
                reconstruct_src[ii],
                reconstruct_dst[ii],
                fake[ii],
            ]
        )
    
    im_grid = (
        torchvision.utils.make_grid(images_for_grid, 3, padding=1)
        .permute(1, 2, 0)
        .cpu()
        .numpy()
    ) * 255
    return im_grid.astype(np.uint8)

def draw_visualization_predicted(
    reconstruct_src,
    target_src,
    reconstruct_dst,
    target_dst,
    fake,
    gap_width=10,
):
    images_for_grid_left = []
    for i in range(5):
        if i < len(target_src):
            images_for_grid_left.extend([
                target_src[i],
                reconstruct_src[i],
                target_dst[i],
                reconstruct_dst[i],
                fake[i],
            ])
    
    images_for_grid_right = []
    for i in range(5, 10):
        if i < len(target_src):
            images_for_grid_right.extend([
                target_src[i],
                reconstruct_src[i],
                target_dst[i],
                reconstruct_dst[i],
                fake[i],
            ])
        else:
            blank = torch.zeros_like(target_src[0])
            images_for_grid_right.extend([blank, blank, blank, blank, blank])
    
    grid_left = torchvision.utils.make_grid(
        images_for_grid_left, 
        nrow=5,
        padding=2,
        normalize=True,
        value_range=(0, 1)
    ).permute(1, 2, 0).cpu().numpy()
    
    grid_right = torchvision.utils.make_grid(
        images_for_grid_right, 
        nrow=5,
        padding=2,
        normalize=True,
        value_range=(0, 1)
    ).permute(1, 2, 0).cpu().numpy()
    
    if grid_left.shape[0] != grid_right.shape[0]:
        max_height = max(grid_left.shape[0], grid_right.shape[0])
        if grid_left.shape[0] < max_height:
            pad_height = max_height - grid_left.shape[0]
            grid_left = np.pad(grid_left, ((0, pad_height), (0, 0), (0, 0)), 'constant')
        if grid_right.shape[0] < max_height:
            pad_height = max_height - grid_right.shape[0]
            grid_right = np.pad(grid_right, ((0, pad_height), (0, 0), (0, 0)), 'constant')
    
    gap = np.ones((grid_left.shape[0], gap_width, 3)) * 0.5
    
    final_grid = np.hstack([grid_left, gap, grid_right])
    
    final_grid = (final_grid * 255).astype(np.uint8)
    
    return final_grid