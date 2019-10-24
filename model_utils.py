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

def exp_lr_scheduler(optimizer, epoch, init_lr=0.0008, lr_decay_epoch=5):
	"""
	Decay learning rate by a factor of 0.1 every lr_decay_epoch epochs.
	
	"""
	lr = init_lr * (0.1**(epoch // lr_decay_epoch))
	print('lr is '+str(lr))

	if (epoch % lr_decay_epoch == 0):
		print('LR is set to {}'.format(lr))

	for param_group in optimizer.param_groups:
		param_group['lr'] = lr

	return optimizer


def init_reg_params(model, freeze_layers = []):
	"""
	Input:
	1) model: A reference to the model that is being trained
	2) freeze_layers: A layer which 

	Output:
	1) reg_params: A dictionary containing importance weights (omega), init_val (keep a reference 
	to the initial values of the parameters) for all trainable parameters


	Function:
	"""
	reg_params = {}

	for name, param in model.named_parameters():
		if not name in freeze_layers:
			print ("Initializing omega values for layer", name)
			omega = torch.FloatTensor(param.size()).zero_()
			init_val = param.data.clone()
			temp = {}

			#for first task, omega is initialized to zero
			temp['omega'] = omega
			temp['init_val'] = init_val

			#the key for this dictionary is the name of the layer
			reg_params[name] = temp

	return reg_params 


def init_reg_params_acrosstasks(model):
	"""
	Input:
	1) model: A reference to the model that is being trained

	Output:
	1) reg_params: A dictionary containing importance weights (omega), init_val (keep a reference 
	to the initial values of the parameters) for all trainable parameters


	Function:
	"""

	#Get the reg_params for the model 
	reg_params = model.reg_params

	for name, param in model.named_parameters():
		
		temp = reg_params[name]
		print ("Initializing the omega values for layer for the new task", name)
		
		#Store the previous values of omega
		prev_omega = temp['omega']
		
		#Initialize a new omega
		new_omega = torch.FloatTensor(param.size()).zero_()
		init_val = param.data.clone()
		
		temp['prev_omega'] = prev_omega
		temp['omega'] = new_omega

		#store the initial values of the parameters
		temp['init_val'] = init_val

		#the key for this dictionary is the name of the layer
		reg_params[name] = temp

	return reg_params


def consolidate_reg_params(model, freeze_layers = []):
	"""
	Input:
	1) model: A reference to the model that is being trained

	Output:
	1) reg_params: A dictionary containing importance weights (omega), init_val (keep a reference 
	to the initial values of the parameters) for all trainable parameters


	Function:
	"""

	#Get the reg_params for the model 
	reg_params = model.reg_params

	for name, param in model.named_parameters():
		
		temp = reg_params[name]
		print ("Consolidating the omega values for layer", name)
		
		#Store the previous values of omega
		prev_omega = temp['prev_omega']
		new_omega = temp['omega']

		new_omega = torch.add(prev_omega, new_omega)
		del temp['prev_omega']
		
		temp['omega'] = new_omega

		#the key for this dictionary is the name of the layer
		reg_params[name] = temp

	return reg_params


def compute_omega_values():
	"""
	global version for computing
	"""

	model.eval()

	for data in dataloader:
		
		#get the inputs and labels
		inputs, labels = data

		if(use_gpu):
			device = torch.device("cuda:0")
			inputs, labels = inputs.to(device), labels.to(device)

		#Zero the parameter gradients
		optimizer.zero_grad()

		#get the function outputs
		outputs = model(inputs)

		l2_norm = torch.norm(outputs, 2)
		squared_l2_norm = l2_norm**2

		#compute gradients for these parameters
		squared_l2_norm.backward()

		print("The batch number is", index)

		optimizer.step(model.reg_params, index, labels.size(0))

		index = index + 1

	return model










