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


def mas_train(no_of_tasks):

	print ("The model is being trained on task {}".format())

	#Need to train over tasks 
	for t in range(no_of_tasks):

		#initialize reg_params for task 0
		if (t == 0):
			model.reg_params = init_reg_params(model, use_gpu)
		
		#initialize reg_params for tasks > 0 
		else:
			model.reg_params = init_reg_params_across_tasks(model, use_gpu)

		train_model()