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


def mas_train(model, task_no, no_of_layers, no_of_classes, dataloader, dset_size, use_gpu = False):
	"""
	Inputs:
	1) model: A reference to the model that is being exposed to the data for the task
	2) task_no: The task that is being exposed to the model identified by it's number
	3) no_of_layers: The number of layers that you want to freeze in the feature extractor of the Alexnet
	4) no_of_classes: The number of classes in the task  
	5) dataloader: Dataloader that feeds data to the model
	6) dset_size: The size of the task (size of the dataset belonging to the task)
	7) use_gpu: Set the flag to `True` if you want to train the model on GPU

	Outputs:
	1) model: Returns a trained model

	Function: Trains the model on a particular task and deals with different tasks in the sequence
	"""

	#this is the task to which the model is exposed
	if (t == 1):
		#initialize the reg_params for this task
		model, freeze_layers = create_freeze_layers(model, no_of_layers)
		model.reg_params = init_reg_params(model, use_gpu, freeze_layers)

	else:
		#inititialize the reg_params for this task
		model.reg_params = init_reg_params_across_tasks(model, use_gpu)

	#get the optimizer
	optimizer_sp = local_sgd(model.tmodel.parameters(), lr = 0.001)

	model = train_model(model, path, optimizer_sp, model_criterion, dataloader, dset_size, num_epochs, checkpoint_file, use_gpu, lr = 0.003)

	if (t > 1):
		model = consolidate_reg_params(model, use_gpu)

	return model



def compute_forgetting(task_no, dataloader, dset_size):
	"""
	Inputs
	1) task_no: The task number on which you want to compute the forgetting 
	2) dataloader: The dataloader that feeds in the data to the model

	Outputs
	1) forgetting: The amount of forgetting undergone by the model

	Function: Computes the "forgetting" that the model has on the 
	"""
	
	#get the results file
	store_path = os.path.join(os.getcwd(), "models", "Task_" + str(task_no))
	model_path = os.path.join(os.getcwd(), "models")

	file_object = open(os.path.join(store_path, "performance.txt"), 'r')
	old_performance = file_object.read()
	file_object.close()

	model = model_inference(task_no, use_gpu = False)

	for data in dataloader:
		inputs, labels = data
		del data

		if (use_gpu):
			input_data = input_data.to(device)
			labels = labels.to(device) 
		
		else:
			input_data  = Variable(input_data)
			labels = Variable(labels)
		
		output = model.tmodel(input_data)
		del input_data

		_, preds = torch.max(outputs, 1)

		loss = model_criterion(output, labels)
		del output
		
		running_loss += loss.item()
		del loss

		running_corrects += torch.sum(preds == labels.data)
		del preds
		del labels

	epoch_loss = running_loss/dset_size
	epoch_accuracy = running_corrects.double()/dset_size

	forgetting = epoch_accuracy - old_performance

	return forgetting