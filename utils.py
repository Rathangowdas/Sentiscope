import os
import random
import logging
import numpy as np
import torch
from typing import Dict

# Global Constants
MODEL_NAME = "bert-base-uncased"
MAX_LENGTH = 128
LABEL_MAP: Dict[str, int] = {"negative": 0, "neutral": 1, "positive": 2}
INV_LABEL_MAP: Dict[int, str] = {0: "negative", 1: "neutral", 2: "positive"}

# ANSI Color Escape Codes
class Colors:
    HEADER = "\033[95m"
    BLUE = "\033[94m"
    CYAN = "\033[96m"
    GREEN = "\033[92m"
    WARNING = "\033[93m"
    FAIL = "\033[91m"
    END = "\033[0m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"


class ColoredFormatter(logging.Formatter):
    """
    Custom logging formatter to output colored logs in the terminal
    based on the log level.
    """
    LEVEL_COLORS = {
        logging.DEBUG: Colors.CYAN,
        logging.INFO: Colors.GREEN,
        logging.WARNING: Colors.WARNING,
        logging.ERROR: Colors.FAIL,
        logging.CRITICAL: Colors.FAIL + Colors.BOLD
    }

    def format(self, record):
        color = self.LEVEL_COLORS.get(record.levelno, Colors.END)
        # Apply color to the levelname
        orig_levelname = record.levelname
        record.levelname = f"{color}{orig_levelname}{Colors.END}"
        
        # Apply color to the message as well for warnings/errors/critical
        if record.levelno >= logging.WARNING:
            orig_msg = record.msg
            record.msg = f"{color}{orig_msg}{Colors.END}"
            
        result = super().format(record)
        
        # Restore original fields just in case
        record.levelname = orig_levelname
        if record.levelno >= logging.WARNING:
            record.msg = orig_msg
            
        return result


def setup_logger(name: str = "SentiScope", log_file: str = "logs/training.log") -> logging.Logger:
    """
    Configures and returns a logger that outputs to both a log file and the console.
    """
    # Create logs directory if it does not exist
    log_dir = os.path.dirname(log_file)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # Avoid duplicate handlers if logger is already configured
    if not logger.handlers:
        # File handler (writes plain text without colors)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_formatter = logging.Formatter(
            "[%(asctime)s] [%(levelname)s] [%(name)s] - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

        # Stream handler (console output with ANSI colors)
        console_handler = logging.StreamHandler()
        console_formatter = ColoredFormatter(
            "[SentiScope] [%(levelname)s] - %(message)s"
        )
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)

    return logger


def set_seed(seed: int = 42) -> None:
    """
    Sets reproducible random seeds for Python, NumPy, and PyTorch.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # Ensure deterministic operations
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    # Simple log statement inside set_seed
    logging.getLogger("SentiScope").debug(f"Reproducibility seed set to {seed}")


def get_device() -> torch.device:
    """
    Automatically detects and returns the best available compute device:
    CUDA, Apple Silicon MPS, or CPU fallback.
    """
    logger = logging.getLogger("SentiScope")
    if torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info(f"Using GPU device: {torch.cuda.get_device_name(0)} (CUDA)")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
        logger.info("Using Apple Silicon GPU device (MPS)")
    else:
        device = torch.device("cpu")
        logger.info("Using CPU fallback device")
    return device


def color_text(text: str, color: str) -> str:
    """
    Utility helper to surround text with ANSI color codes.
    """
    return f"{color}{text}{Colors.END}"
