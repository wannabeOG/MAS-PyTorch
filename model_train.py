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

sys.path.append('utils')
from model_utils import *
from mas_utils import *


train_model(model, path, optimizer_sp, model_criterion, dataloader, dset_size, num_epochs, checkpoint_file, use_gpu, lr = 0.003)
def train_model(model, path, optimizer, model_criterion, dset_loaders, dset_size, num_epochs, checkpoint_file, use_gpu, lr = 0.003):
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
			optimizer_ft = omega_update(model.reg_params)
			print ("Updating the omega values for this task")
			model = compute_omega_grads_norm(model, dataloader, optimizer_ft)
		
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

train_model(model, path, optimizer_sp, model_criterion, dataloader, dset_size, num_epochs, checkpoint_file, use_gpu, lr = 0.003)
def train_model(model, path, optimizer_sp, model_criterion, dset_loaders, dset_size, num_epochs, use_gpu, lr = 0.1):
	""" 
	Inputs: 
		1) num_classes = The number of classes in the new task  
		2) feature_extractor = A reference to the feature extractor model  
		3) encoder_criterion = The loss criterion for training the Autoencoder
		4) dset_loaders = Dataset loaders for the model
		5) dset_size = Size of the dataset loaders
		6) num_of_epochs = Number of epochs for which the model needs to be trained
		7) use_gpu = A flag which would be set if the user has a CUDA enabled device
		8) task_number = A number which represents the task for which the model is being trained
		9) lr = initial learning rate for the model
		10) alpha = Tradeoff factor for the loss   

	Function: Trains the model on the given task
		1) If the task relatedness is greater than 0.85, the function uses the Learning without Forgetting method
		2) If the task relatedness is lesser than 0.85, the function uses the normal finetuning procedure as outlined
			in the "Learning without Forgetting" paper ("https://arxiv.org/abs/1606.09282")

		Whilst implementing finetuning procedure, PyTorch does not provide the option to only partially freeze the 
		weights of a layer. In order to implement this idea, I manually zero the gradients from the older classes in
		order to ensure that these weights do not have a learning signal from the loss function. 

	"""	
	
	device = torch.device("cuda:0" if use_gpu else "cpu") 
	
	# Load the most related model in the memory and finetune the model
	new_path = os.getcwd() + "/models/trained_models"
	path = os.getcwd() + "/models/trained_models/model_"
	path_to_dir = path + str(model_number) 
	file_name = path_to_dir + "/classes.txt" 
	file_object = open(file_name, 'r')
	
	num_of_classes_old = file_object.read()
	file_object.close()
	num_of_classes_old = int(num_of_classes_old)

	#Create a variable to store the new number of classes that this model is exposed to
	new_classes = num_of_classes_old + num_classes
	
	#Check the number of models that already exist

	num_ae = len(next(os.walk(new_path))[1])

	#If task_number is less than num_ae it suggests that the directory had already been created
	if (task_number <= num_ae):
		#Keeping it consistent with the usage of num_ae throughout this file
		num_ae = task_number-1

	
	print ("Checking if a prior training file exists")
	
	#mypath is the path where the model is going to be stored
	mypath = path + str(num_ae+1)

	#The conditional if the directory already exists
	if os.path.isdir(mypath):
		#mypath = path + str(num_ae+1)

		######################### check for the latest checkpoint file #######################
		onlyfiles = [f for f in os.listdir(mypath) if os.isfile(os.join(mypath, f))]
		max_train = -1
		flag = False

		#Check the latest epoch file that was created
		for file in onlyfiles:
			if(file.endswith('pth.tr')):
				flag = True
				test_epoch = file[0]
				if(test_epoch > max_train): 
					max_epoch = test_epoch
					checkpoint_file = file
		#######################################################################################
		
		if (flag == False): 
			checkpoint_file = ""

		
		#Steps to create a ref_model in order to prevent storing this model as well
		model_init = GeneralModelClass(num_of_classes_old)
		model_init.load_state_dict(torch.load(path_to_dir+"/best_performing_model.pth"))
		
		#Create (Recreate) the ref_model that has to be used
		ref_model = copy.deepcopy(model_init)
		ref_model.train(False)
		ref_model.to(device)
		del model_init

		######################## Code for loading the checkpoint file #########################
		
		if (os.path.isfile(mypath + "/" + checkpoint_file)):
			
			print ("Loading checkpoint '{}' ".format(checkpoint_file))
			checkpoint = torch.load(checkpoint_file)
			start_epoch = checkpoint['epoch']
			
			print ("Loading the model")
			model_init = GeneralModelClass(num_of_classes_old + num_classes)
			model_init = model_init.load_state_dict(checkpoint['state_dict'])
			
			print ("Loading the optimizer")
			optimizer = optimizer.load_state_dict(checkpoint['optimizer'])
			
			print ("Done")

		else:
			start_epoch = 0

		##########################################################################################

	#Will have to create a new directory since it does not exist at the moment
	else:
		print ("Creating the directory for the new model")
		os.mkdir(mypath)


	# Store the number of classes in the file for future use
		with open(os.path.join(mypath, 'classes.txt'), 'w') as file1:
			input_to_txtfile = str(new_classes)
			file1.write(input_to_txtfile)
			file1.close()

	# Load the most related model into memory
	
		print ("Loading the most related model")
		model_init.load_state_dict(torch.load(path_to_dir+"/best_performing_model.pth"))
		print ("Model loaded")

		for param in model_init.Tmodel.classifier.parameters():
			param.requires_grad = True

		for param in model_init.Tmodel.features.parameters():
			param.requires_grad = False

		for param in model_init.Tmodel.features[8].parameters():
			param.requires_grad = True

		for param in model_init.Tmodel.features[10].parameters():
			param.requires_grad = True

		
		#model_init.to(device)
		print ("Initializing an Adam optimizer")
		optimizer = optim.Adam(model_init.Tmodel.parameters(), lr = 0.003, weight_decay= 0.0001)


		# Reference model to compute the soft scores for the LwF(Learning without Forgetting) method
		
		
		#Actually makes the changes to the model_init, so slightly redundant
		print ("Initializing the model to be trained")
		model_init = initialize_new_model(model_init, num_classes, num_of_classes_old)
		model_init.to(device)
		start_epoch = 0

	#The training process format or LwF (Learning without Forgetting)
	# Add the start epoch code 
	
	if (best_relatedness > 0.85):

		print ("Using the LwF approach")
		for epoch in range(start_epoch, num_epochs):			
			since = time.time()
			best_perform = 10e6
			
			print ("Epoch {}/{}".format(epoch+1, num_epochs))
			print ("-"*20)
			print ("The training phase is ongoing".format(phase))
			
			running_loss = 0
			
			#scales the optimizer every 10 epochs 
			optimizer = exp_lr_scheduler(optimizer, epoch, lr)
			model_init = model_init.train(True)
			
			for data in dset_loaders:
				input_data, labels = data

				del data

				if (use_gpu):
					input_data = Variable(input_data.to(device))
					labels = Variable(labels.to(device)) 
				
				else:
					input_data  = Variable(input_data)
					labels = Variable(labels)
				
				model_init.to(device)
				ref_model.to(device)
				
				output = model_init(input_data)
				ref_output = ref_model(input_data)

				del input_data

				optimizer.zero_grad()
				model_init.zero_grad()

				# loss_1 only takes in the outputs from the nodes of the old classes 

				loss1_output = output[:, :num_of_classes_old]
				loss2_output = output[:, num_of_classes_old:]

				del output

				loss_1 = model_criterion(loss1_output, ref_output, flag = "Distill")
				
				del ref_output
				
				# loss_2 takes in the outputs from the nodes that were initialized for the new task
				
				loss_2 = model_criterion(loss2_output, labels, flag = "CE")
				
				del labels
				#del output

				total_loss = alpha*loss_1 + loss_2

				del loss_1
				del loss_2

				
				total_loss.backward()
				optimizer.step()

				running_loss += total_loss.item()
				
			epoch_loss = running_loss/dset_size


			print('Epoch Loss:{}'.format(epoch_loss))

			if(epoch != 0 and epoch != num_of_epochs -1 and (epoch+1) % 10 == 0):
				epoch_file_name = os.path.join(mypath, str(epoch+1)+'.pth.tar')
				torch.save({
				'epoch': epoch,
				'epoch_loss': epoch_loss, 
				'model_state_dict': model_init.state_dict(),
				'optimizer_state_dict': optimizer.state_dict(),

				}, epoch_file_name)


		torch.save(model_init.state_dict(), mypath + "/best_performing_model.pth")		
		

		del model_init
		del ref_model



def train_model():

	omega_epochs = num_epochs + 1

	####################### Check if the directory exists ###############################
	store_path = os.path.join(os.getcwd(), "models", "Task_" + str(task_no))
	checkpoint_file = check_checkpoint(store_path)

	
	if (checkpoint_file == ""):
		start_epoch = 0

	else:
		print ("Loading checkpoint '{}' ".format(checkpoint_file))
		checkpoint = torch.load(checkpoint_file)
		start_epoch = checkpoint['epoch']
		
		print ("Loading the model")
		model = shared_model(models.alexnet(pretrained = True))
		model = model.load_state_dict(checkpoint['state_dict'])
		
		print ("Loading the optimizer")
		optimizer = local_sgd(model.reg_params)
		optimizer = optimizer.load_state_dict(checkpoint['optimizer'])
		
		print ("Done")

	#######################################################################################
	

	for epoch in range(start_epoch, omega_epochs):			
		
		if (epoch == omega_epochs -1):
			optimizer_ft = omega_update(model.reg_params)
			print ("Updating the omega values for this task")
			model = compute_omega_grads_norm(model, dataloader, optimizer_ft)
			return

		since = time.time()
		best_perform = 10e6

		print ("Epoch {}/{}".format(epoch+1, num_epochs))
		print ("-"*20)
		print ("The {}ing phase is ongoing".format(phase))
		
		running_loss = 0
		
		#scales the optimizer every 10 epochs 
		optimizer = exp_lr_scheduler(optimizer, epoch, lr)

		if(phase == 'train'):
			model.train(True)

		else:
			model.eval()

		for data in dset_loaders[phase]:
			input_data, labels = data

			del data

			if (use_gpu):
				input_data = Variable(input_data.to(device))
				labels = Variable(labels.to(device)) 
			
			else:
				input_data  = Variable(input_data)
				labels = Variable(labels)
			
			model.to(device)
			optimizer.zero_grad()
			
			output = model.tmodel(input_data)
			_, preds = torch.max(outputs, 1)

			del input_data

			loss = model_criterion(output, labels)
			
			loss.backward()
			optimizer.step(model.reg_params)
		
			else:

			running_loss += loss.item()
			running_corrects += torch.sum(preds == labels.data)
			
		epoch_loss = running_loss/dset_size
		epoch_accuracy = running_corrects.double()/dset_size


		print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))
		
		if(epoch != 0 and epoch != num_of_epochs -1 and (epoch+1) % 10 == 0):
			epoch_file_name = os.path.join(mypath, str(epoch+1)+'.pth.tar')
			torch.save({
			'epoch': epoch,
			'epoch_loss': epoch_loss, 
			'model_state_dict': model_init.state_dict(),
			'optimizer_state_dict': optimizer.state_dict(),

			}, epoch_file_name)


	save_model(model, task_no)
