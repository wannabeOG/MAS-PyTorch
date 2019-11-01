#!/usr/bin/env python
# coding: utf-8

import warnings
import os
from pathlib import Path
import shutil

import requests, zipfile, io

import sys
sys.path.append(os.path.join(os.getcwd(), 'utils'))


def create_val_img_folder(dataset_dir):
	'''
	This method is responsible for separating validation images into separate sub folders, so that 
	Pytorch's ImageFolder can be used to prepare the dataloaders for the test set
	'''

	#validation directory
	val_dir = dataset_dir
	img_dir = os.path.join(val_dir, 'images')

	#Open the file to read off the labels for the different images in the folder
	fp = open(os.path.join(val_dir, 'val_annotations.txt'), 'r')
	data = fp.readlines()

	#Create a dictionary to store this information
	val_img_dict = {}
	
	for line in data:
		words = line.split('\t')
		val_img_dict[words[0]] = words[1]
	fp.close()

	#Create a folder if not present and move these images into proper folders
	for img, folder in val_img_dict.items():
		newpath = (os.path.join(val_dir, folder))
		
		#Create this directory if this does not exist
		if not os.path.exists(newpath):
			os.makedirs(newpath)
		
		#Shift these images into the folder
		if os.path.exists(os.path.join(img_dir, img)):
			os.rename(os.path.join(img_dir, img), os.path.join(newpath, img))

	shutil.rmtree(os.path.join(val_dir, 'images'))
	os.remove(os.path.join(val_dir, 'val_annotations.txt'))
	

def convert_tiny_imagenet(path):
	"""
	Called for train/val/test
	"""

	path = Path(path)
	
	#Delete the file (bounding boxes coordinates et al)
	if not os.path.isdir(path):
		os.remove(path)
		return 

	#Execute in the case of a directory	
	for directory in os.listdir(path):
		if (path.name == 'train'):
			path_to_dir = os.path.join(path, directory)
			dest_path = path_to_dir

			for file in os.listdir(os.path.join(path_to_dir,  'images')):
				shutil.move(path_to_dir + "/" + "images" + "/" + file, dest_path + "/" + file)

			os.rmdir(path_to_dir + "/" + "images")

			for file in os.listdir(path_to_dir):
				if file.endswith('.txt'):
					os.remove(path_to_dir + "/" + file)

		elif (path.name == 'val'):
			if(os.path.isdir(os.path.join(path, directory))): 
				create_val_img_folder(path)

		else:
			if os.path.isdir(path):
				shutil.rmtree(path)
			else:
				os.remove(path)

def convert_to_tasks(path, number_of_tasks):
	"""
	This function converts the dataset into 4 tasks with 50 classes each. Each Task has a seperate
	"training" and "test" folders for carrying out this evaluation function
	"""	
	
	source_train_path = os.path.join(path, "tiny-imagenet-200", "train")
	source_test_path = os.path.join(path, "tiny-imagenet-200", "val")
	
	list_dir_train = os.listdir(source_train_path)
	list_dir_test = os.listdir(source_test_path)
	
	for i in range(number_of_tasks):
		#Create a task directory
		target_path = os.path.join(path, "Task_" + str(i+1)) 
		os.mkdir(target_path)

		#Create a train and a test directory
		target_path_train = os.path.join(target_path, "train")
		target_path_test = os.path.join(target_path, "test")
		
		os.mkdir(target_path_train)
		os.mkdir(target_path_test)
		
		
		for i in range(50):
			a = list_dir_train.pop(0)
			b = list_dir_test.pop(0)

			shutil.move(os.path.join(source_train_path, a), os.path.join(target_path_train, a))
			shutil.move(os.path.join(source_test_path, b), os.path.join(target_path_test, b))


	shutil.rmtree(os.path.join(path, 'tiny-imagenet-200'))


#Create the Data directory 
path_to_file = os.getcwd() + "/Data"
os.mkdir(path_to_file)

#Code to download the dataset to the folder
zip_file_url = 'http://cs231n.stanford.edu/tiny-imagenet-200.zip' 
r = requests.get(zip_file_url)

with zipfile.ZipFile(io.BytesIO(r.content), 'r') as zip_ref:
	zip_ref.extractall(path_to_file)

path_to_dataset = path_to_file + "/tiny-imagenet-200"

#Prep the Data now
file_list = ['train', 'test', 'val', 'wnids.txt', 'words.txt']

for file in file_list:
	convert_tiny_imagenet(path_to_dataset + "/" + file)

#Divide the dataset into 4 tasks
convert_to_tasks(path_to_file, 4)

