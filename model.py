from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np
import torchvision
from torchvision import datasets, models, transforms

import copy
import os
import shutil

from model_utils import *
from model_train import *
from optimizer_lib import *

#The idea is to have classification layers for different tasks


#class specific features are only limited to the last linear layer of the model
class classification_head(nn.Module):
	"""
	
	Each task has a seperate classification head which houses the features that
	are specific to that particular task. These features are unshared across tasks
	as described in section 5.1 of the paper

	"""
	
	def __init__(self, in_features, out_features):
		super(Classification_head, self).__init__()
		self.fc1 = nn.Linear(in_features, out_features)

	def forward(self, x):
		return x




