import torch
import os
from time import time
import random
import numpy as np


class pipeline_step:
    def __init__(self, name: str, speak: bool = True):
        self.name = name
        self.start_time = None
        self.end_time = None
        self.duration = None
        self.speak = speak

    def __enter__(self):
        self.start_time = time()
        if self.speak:
            pipeline_step.print_full_width("-", "-")
            pipeline_step.print_full_width(" ", " ")
            pipeline_step.print_full_width("=", "=")
            opening_message = f"Starting {self.name}"
            pipeline_step.print_full_width(opening_message, "")
            pipeline_step.print_full_width(". " * (len(opening_message) // 2), " ")

    def __exit__(self, exc_type, exc_val, exc_tb):
        torch.cuda.empty_cache()
        self.end_time = time()
        self.duration = self.end_time - self.start_time
        if self.speak:
            closing_message = f"{self.name} completed in {self.duration:.2f} seconds."
            pipeline_step.print_full_width(". " * (len(closing_message) // 2), "")
            pipeline_step.print_full_width(closing_message, "")
            pipeline_step.print_full_width("-", "-")
        return False

    @staticmethod
    def print_full_width(text: str, filler_char: str = " ") -> None:
        terminal_width = os.get_terminal_size().columns
        print(f"{text:{filler_char}^{terminal_width}}")


def seed_all(seed):
    # Set seed for Python random number generator
    random.seed(seed)

    # Set seed for NumPy
    np.random.seed(seed)

    # Set seed for PyTorch CPU
    torch.manual_seed(seed)

    # Set seed for PyTorch GPU (if available)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # For multi-GPU setups
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
