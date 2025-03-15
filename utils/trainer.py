class Trainer(object):
    """
    Base class for trainers, providing basic interface definitions
    """
    
    def __init__(self, config):
        """
        Initializes the trainer
        
        Args:
            config: Configuration object
        """
        self.cfg = config
    
    def setup_models(self):
        """Sets up models, must be implemented by subclasses"""
        raise NotImplementedError
    
    def setup_dataloaders(self):
        """Sets up data loaders, must be implemented by subclasses"""
        raise NotImplementedError
    
    def setup_training(self):
        """Sets up training parameters (optimizer, scheduler, etc.), must be implemented by subclasses"""
        raise NotImplementedError
    
    def load_checkpoint(self):
        """Loads checkpoint, must be implemented by subclasses"""
        raise NotImplementedError
    
    def save_checkpoint(self):
        """Saves checkpoint, must be implemented by subclasses"""
        raise NotImplementedError
    
    def train_step(self, *args, **kwargs):
        """Executes a single training step, must be implemented by subclasses"""
        raise NotImplementedError
    
    def fit(self):
        """Training main loop, must be implemented by subclasses"""
        raise NotImplementedError