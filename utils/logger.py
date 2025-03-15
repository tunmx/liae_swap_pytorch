import sys
from loguru import logger

class SimpleLogger:
    def __init__(self, name, filename="training.log"):
        self.name = name
        self.filename = filename
        
        logger.remove()
        
        logger.add(
            self.filename,
            format="{time:YYYY-MM-DD HH:mm:ss} - {level} - {message}",
            level="INFO",
            mode="a"
        )
        
        logger.add(
            sys.stdout,
            format="{time:YYYY-MM-DD HH:mm:ss} - {level} - {message}",
            level="INFO"
        )
    
    def info(self, msg):
        logger.info(f"[{self.name}] {msg}")
    
    def err(self, msg):
        logger.error(f"[{self.name}] {msg}")
    
    def warn(self, msg):
        logger.warning(f"[{self.name}] {msg}")
