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

	omega_epochs = num_epochs + 1

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

	for epoch in range(start_epoch, omega_epochs):

		if (epoch == omega_epochs -1):
			
			print ("Updating the omega values for this task")
			compute_omega_grads_norm(model, dataloader, optimizer, )
		
		else:

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
				
				model = model.to(device)

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
					
					outputs = model(input_data)
					_, preds = torch.max(outputs.data, 1)
					
					loss = model_criterion(preds, labels)

					if (phase == 'train'):
						loss.backward()
						optimizer.step(model.reg_params)


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
