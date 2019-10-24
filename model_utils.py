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
			reg_params[param] = temp

	return reg_params 


def init_reg_params_across_tasks(model):
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
		reg_params[param] = temp

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
		reg_params[param] = temp

	return reg_params


def compute_omega_grads_norm():
	"""
	global version for computing
	"""

	model.eval()

	index = 0
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

		#compute the sqaured l2 norm of the function outputs
		l2_norm = torch.norm(outputs, 2)
		squared_l2_norm = l2_norm**2

		#compute gradients for these parameters
		squared_l2_norm.backward()

		optimizer.step(model.reg_params, index, labels.size(0))

		index = index + 1

	return model

def compute_omega_grads_vector():
	"""
	global version for computing
	"""

	model.eval()

	index = 0

	for dataloader in dset_loaders:
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

			for unit_no in range(0, outputs.size(1)):
				ith_node = outputs[:, unit_no]
				targets = torch.sum(ith_node)

				#final node in the layer
				if(node_no == outputs.size(1)-1):
					targets.backward()
				else:
					#This retains the computational graph for further computations 
					targets.backward(retain_graph = True)

				optimizer.step(model.reg_params, True, index, labels.size(0))
				optimizer.zero_grad()

			
		optimizer.step(model.reg_params, False, index, labels.size(0))
		index = index + 1

	return model


def model_criterion():
	loss =  nn.CrossEntropyLoss()
	return loss(preds, labels)




def train_model(model, feature_extractor, path, optimizer, encoder_criterion, dset_loaders, dset_size, num_epochs, checkpoint_file, use_gpu, lr = 0.003):
	"""
	Inputs:
		1) model = A reference to the Autoencoder model that needs to be trained 
		2) feature_extractor = A reference to to the feature_extractor part of Alexnet; it returns the features
		   from the last convolutional layer of the Alexnet
		3) path = The path where the model will be stored
		4) optimizer = The optimizer to optimize the parameters of the Autoencoder
		5) encoder_criterion = The loss criterion for training the Autoencoder
		6) dset_loaders = Dataset loaders for the model
		7) dset_size = Size of the dataset loaders
		8) num_of_epochs = Number of epochs for which the model needs to be trained
		9) checkpoint_file = A checkpoint file which can be used to resume training; starting from the epoch at 
		   which the checkpoint file was created 
		10) use_gpu = A flag which would be set if the user has a CUDA enabled device 

	Function:
		Returns a trained autoencoder model

	"""
	since = time.time()
	best_perform = 10e6
	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	num_of_classes = 0

	######################## Code for loading the checkpoint file #########################
	
	if (os.path.isfile(path + "/" + checkpoint_file)):
		path_to_file = path + "/" + checkpoint_file
		print ("Loading checkpoint '{}' ".format(checkpoint_file))
		checkpoint = torch.load(checkpoint_file)
		start_epoch = checkpoint['epoch']
		model = model.load_state_dict(checkpoint['state_dict'])
		print ("Loading the optimizer")
		optimizer = optimizer.load_state_dict(checkpoint['optimizer'])
		print ("Done")

	else:
		start_epoch = 0

	##########################################################################################

	for epoch in range(start_epoch, num_epochs):

		print ("Epoch {}/{}".format(epoch+1, num_epochs))
		print ("-"*10)

		# The model is evaluated at each epoch and the best performing model 
		# on the validation set is saved 

		for phase in ['train', 'val']:

			if (phase == 'train'):
				optimizer = exp_lr_scheduler(optimizer, epoch, lr)
				model.train(True)

			else:
				model.train(False)
				model.eval(True)
			
			running_loss = 0
			
			for data in dset_loaders[phase]:
				
				input_data, labels = data

				del labels
				del data

				if (use_gpu):
					input_data = Variable(input_data.to(device))
					labels = Variable(labels.to(device)) 
				
				else:
					input_data  = Variable(input_data)
					labels = Variable(labels)

				# Input_to_ae is the features from the last convolutional layer
				# of an Alexnet trained on Imagenet 

				#input_data = F.sigmoid(input_data)
				
				optimizer.zero_grad()
				
				model.to(device)

				outputs = model(input_data)
				_, preds = torch.max(outputs.data, 1)
				loss = model_criterion(preds, labels)

				if (phase == 'train'):
					loss.backward()
					optimizer.step()


				running_loss += loss.item()
			
			epoch_loss = running_loss/dset_size

			
			print('Epoch Loss:{}'.format(epoch_loss))
				
			#Creates a checkpoint every 5 epochs
			if(epoch != 0 and (epoch+1) % 5 == 0 and epoch != num_of_epochs - 1):
				epoch_file_name = os.path.join(path, str(epoch+1)+'.pth.tar')
				torch.save({
				'epoch': epoch,
				'epoch_loss': epoch_loss, 
				'model_state_dict': model.state_dict(),
				'optimizer_state_dict': optimizer.state_dict(),

				}, epoch_file_name)


		
	torch.save(model.state_dict(), path + "/best_performing_model.pth")

	elapsed_time = time.time()-since
	print ("This procedure took {:.2f} minutes and {:.2f} seconds".format(elapsed_time//60, elapsed_time%60))
	print ("The best performing model has a {:.2f} loss on the test set".format(best_perform))



