import torch
import random
import os
import numpy as np
import torch
from torch.optim.lr_scheduler import _LRScheduler

def init_seed(seed=42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

class WarmupCosineScheduler(_LRScheduler):
    """
    Learning rate scheduler that implements warmup and cosine annealing
    
    Features:
    - Linear warmup phase
    - Optional learning rate dropout after warmup
    - Cosine annealing schedule after warmup (if lr_dropout is True)
    """

    def __init__(
        self,
        optimizer,
        max_lr,          # Maximum learning rate after warmup
        total_steps,     # Total number of training steps
        warmup_steps,    # Number of warmup steps
        lr_dropout,      # Whether to use learning rate dropout
        lr_cos,         # Number of steps for cosine annealing
        eta_min=0,      # Minimum learning rate for cosine annealing
        last_epoch=-1,
    ):
        self.max_lr = max_lr
        self.total_steps = total_steps
        self.warmup_steps = warmup_steps
        self.lr_dropout = lr_dropout
        self.lr_cos = lr_cos
        self.eta_min = eta_min
        self.current_step = 0
        
        if self.lr_dropout:
            self.cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=self.lr_cos,
                eta_min=self.eta_min
            )
        
        super(WarmupCosineScheduler, self).__init__(optimizer, last_epoch)
    
    def get_lr(self):
        """
        Calculate the learning rate based on current step
        Returns:
            list: Learning rates for each parameter group
        """
        # During warmup phase, linearly increase learning rate
        if self.current_step < self.warmup_steps:
            return [base_lr * (self.current_step / self.warmup_steps) for base_lr in self.base_lrs]
        
        # After warmup phase
        if self.lr_dropout:
            # If using lr_dropout, return the current learning rates from cosine scheduler
            return [group['lr'] for group in self.cosine_scheduler.optimizer.param_groups]
        
        # If not using lr_dropout, maintain constant learning rate at max_lr
        return self.base_lrs
    
    def step(self, step=None):
        """
        Update scheduler state and learning rate
        Args:
            step (int, optional): Manually specify current step
        Returns:
            float: Current learning rate
        """
        if step is not None:
            self.current_step = step
        else:
            self.current_step += 1
            
        if self.current_step < self.warmup_steps or not self.lr_dropout:
            super(WarmupCosineScheduler, self).step()
        elif self.lr_dropout:
            if self.current_step == self.warmup_steps:
                for param_group, lr in zip(self.optimizer.param_groups, self.base_lrs):
                    param_group['lr'] = lr
            
            self.cosine_scheduler.step()
        
        return self.optimizer.param_groups[0]["lr"]
    
    def state_dict(self):
        """
        Returns the state of the scheduler as a dict
        Used for checkpointing
        """
        state_dict = {
            'base_lrs': self.base_lrs,
            'current_step': self.current_step
        }
        
        if self.lr_dropout:
            state_dict['cosine_scheduler'] = self.cosine_scheduler.state_dict()
            
        return state_dict
    
    def load_state_dict(self, state_dict):
        """
        Loads scheduler state
        Args:
            state_dict (dict): Scheduler state
        """
        self.base_lrs = state_dict['base_lrs']
        self.current_step = state_dict['current_step']
        
        if self.lr_dropout and 'cosine_scheduler' in state_dict:
            self.cosine_scheduler.load_state_dict(state_dict['cosine_scheduler'])

