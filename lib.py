import os
import os.path as osp

import random
import xml.etree.ElementTree as ET
import cv2
import torch.utils.data as data
import numpy as np
import torch
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.functional as F
from torch.autograd import Function
torch.manual_seed(1234)
np.random.seed(1234)
random.seed(1234)