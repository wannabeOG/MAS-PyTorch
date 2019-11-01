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

import sys

sys.path.append('utils')
from model_utils import *
from mas_utils import *

from optimizer_lib import *
from model_train import *


def mas_train(model, task_no, path_to_datasets, use_gpu = False):

	#this is the task to which the model is exposed
	if (t == 1):
		#initialize the reg_params for this task
		model.reg_params = init_reg_params(model, use_gpu)

	else:
		#inititialize the reg_params for this task
		model.reg_params = init_reg_params_across_tasks(model, use_gpu)

	#get the optimizer
	optimizer_sp = local_sgd()

def mas_test():
