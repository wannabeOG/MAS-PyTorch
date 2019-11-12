#!/usr/bin/env python
# coding: utf-8

from __future__ import print_function

import torch
import torch.nn as nn
import torchvision
from torchvision import datasets, models, transforms

import os
import shutil

#The idea is to have classification layers for different tasks


#class specific features are only limited to the last linear layer of the model
class classification_head(nn.Module):
	"""
	
	Each task has a seperate classification head which houses the features that
	are specific to that particular task. These features are unshared across tasks
	as described in section 5.1 of the paper

	"""
	
	def __init__(self, in_features, out_features):
		super(classification_head, self).__init__()
		self.fc = nn.Linear(in_features, out_features)

	def forward(self, x):
		return x
		

class shared_model(nn.Module):

	def __init__(self, model):
		super(shared_model, self).__init__()
		self.tmodel = models.alexnet(pretrained = True)
		self.reg_params = {}

	def forward(self, x):
		return self.tmodel(x)
		