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



def model_criterion():
	loss =  nn.CrossEntropyLoss()
	return loss(preds, labels)


def create_task_dir(task_no, no_of_classes):
	"""
	Inputs
	1) task_no: The identity for the task defined by it's number in the sequence
	2) no_of_classes: The number of classes that the particular task has 

	Outputs
	1) store_path: A string which represents the directory where the classification head will be stored

	Function: This function creates a directory to store the classification head for the new task. It also 
	creates a text file which stores the number of classes that this task contained
	
	"""

	curr_dir = os.getcwd()
	
	if not (os.isdir(os.path.join(curr_dir, "models"))):
		os.mkdir(os.path.join(curr_dir, "models"))

	store_path = os.path.join(curr_dir, "models", "Task_" + str(task_no))
	os.mkdir(store_path)

	file_path = os.path.join(store_path, "classes.txt") 

	with open(file_path, 'w') as file1:
		input_to_txtfile = str(no_of_classes)
		file1.write(input_to_txtfile)
		file1.close()
	
	return store_path


def model_inference(task_no, use_gpu = False):
	"""
	Inputs
	1) task_no: The task number for which the model is being evaluated
	2) use_gpu: Set the flag to True if you want to run the code on GPU. Default value: False

	Outputs
	1) model: A reference to the model

	Function: Combines the classification head for a particular task with the shared model and
	returns a reference to the model is used for testing the process

	"""

	#all models are derived from an alexnet architecture
	model = models.alexnet(pretrained = True)
	path_to_model = os.path.join(os.getcwd(), "models")

	#load the trained shared model
	model.load_state_dict(torch.load(path_to_model))

	path_to_head = os.path.join(os.getcwd(), "models", "Task_" + str(task_no))
	
	#get the number of classes by reading from the text file created during initialization for this task
	file_name = os.path.join(path_to_head, "classes.txt") 
	file_object = open(file_name, 'r')
	file_object.close()
	
	num_classes = int(num_classes)
	in_features = model.classfier[-1].in_features
	
	#load the classifier head for the given task identified by the task number
	classifier = classification_head(in_features, num_classes)
	classifier.load_state_dict(torch.load(os.path.join(path_to_head, "head.pth")))

	#change the weights layers to the classifier head weights
	model.classifier[count-1].weight.data = classifier.fc.weight.data
	model.classifier[count-1].bias.data = classifier.fc.bias.data

	device = torch.device("cuda:0" if use_gpu else "cpu")
	model.eval()
	model.to(device)
	
	return model



def model_init(no_classes, use_gpu = False):
	"""
	Inputs
	1) no_classes: The number of classes that the model is exposed to in the new task
	2) use_gpu: Set the flag to True if you want to run the code on GPU. Default value = False

	Outputs
	1) model: A reference to the model that has been initialized

	Function: Initializes a model for the new task which the shared features and a classification head
	particular to the new task

	"""

	path = os.path.join(os.getcwd(), "models", "shared_model.pth")
	model = models.alexnet(pretrained = True)
	
	if os.path.isfile(path):
		model.load_state_dict(torch.load(path))

	#initialize a new classification head
	model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, dset_classes)

	device = torch.device("cuda:0" if use_gpu else "cpu")
	model.train(True)
	model.to(device)

	return model



def save_model(model, task_no):
	"""
	Inputs
	1) model: A reference to the model that needs to be saved
	2) task_no: The task number identifies the task for which the model is to be saved

	Function: Saves a reference for the classification head and the shared model at the 
	appropriate locations

	"""

	in_features = model.classifier[-1].in_features 
	out_features = model.classifier[-1].out_features

	#seperate out the classification head from the model
	ref = classification_head(in_features, out_features)
	ref.fc1.weight.data = model.classifier[-1].weight.data
	ref.fc1.bias.data = model.classifier[-1].bias.data

	#hacky fix for storing the shared model
	path = os.path.join(os.getcwd(), "models", "shared_model.pth")
	torch.save(model.state_dict(), path)
	del path
	del model

	store_path = create_task_dir(task_no, no_of_classes)
	torch.save(ref.state_dict(), os.path.join(store_path, "head.pth"))

	del ref