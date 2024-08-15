"""Top-level package for Stress-in-Action (SiA)."""

import warnings
warnings.filterwarnings("ignore")

# Dependencies
import numpy as np
import datasets

# Info
__version__ = "1.0.0"

# Maintainer info
__author__ = "Alex Antonides"

# Subpackages
from .builders import *

warnings.filterwarnings("default")