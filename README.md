Memory Aware Synapses: Learning what (not) to forget
========================================

Code for the Paper:

**[Memory Aware Synapses: Learning what (not) to forget][10]**\
Rahaf Aljundi, Francesca Babiloni, Mohamed Elhoseiny, Marcus Rohrbach, Tinne Tuytelaars\
[ECCV 2018]

If you find this code useful, please consider citing the original work by authors:

```
@InProceedings{Aljundi_2018_ECCV,
author = {Aljundi, Rahaf and Babiloni, Francesca and Elhoseiny, Mohamed and Rohrbach, Marcus and Tuytelaars, Tinne},
title = {Memory Aware Synapses: Learning what (not) to forget },
booktitle = {The European Conference on Computer Vision (ECCV)},
month = {September},
year = {2018}
}
```

Introduction
---------------------------

Lifelong Machine Learning, or LML, considers systems that can learn many tasks over a lifetime from one or more domains. They retain the knowledge they have learned and use that knowledge to more efficiently and effectively learn new tasks more effectively and efficiently (This is a case of positive inductive bias where the past knoweledge helps the model to perform better on the newer task). In the case of continual learning, one of the
key constraints is that the data belonging to the previous tasks cannot be stored. This may be either due to privacy concerns or memory limitations. This is one of the primary differences between the paradigms of Multi Task learning and Continual Learning 

The problem of Catastrophic Inference or Catstrophic Forgetting is one of the major hurdles facing this domain where the performance of the model inexplicably declines on the older tasks once the newer tasks are introduced into the learning pipeline. 

The approaches prevalent in literature at the moment can be sub divided into the following [two categories][1]:
1) Prior focussed: The prior focussed approaches use a penalty term to regularize the parameters rather than
a hard constraint   
2) Parameter Isolation: This approach reserves different parameters for different tasks to prevent interference  
3) Replay-based approach: This approach is similar to experience replay from Reinforcment Learning wherein certain examples are stored in a buffer which is then used to stablize the training of a shared model. 

This paper belongs to the first approach. It derives it's inspiration from the [Hebbian learning theory][2] which can be insufficiently summarized as "Synapses that fire together learn together". This paper has a similar idea to [Elastic Weight Consolidation][3]. To offset the memory limitations of this approach, this paper tries to determine an importance weight for each of the model parameters. These importance weights are then stored in conjunction with the model parameters. The loss function for such an approach comprises of two parts, the first term is the traditional cross entropy loss and the second term is a penalty for changes to weights of the network; a penalty term that is proportional to the importance weight of the parameter.


Requisites
-----------------------------

* PyTorch
  Use the instructions that are outlined on [PyTorch Homepage][4] for installing PyTorch for your operating system
* Python 3.6


<a name="someid"></a> Datasets and Designing the experiments
----------------------------------------------------------------

The original paper uses [Caltech-UCSD Birds][5], [MIT Scenes][6] and [Oxford Flowers][7]. Compuatational and hardware limitations necessitated the design of experiments such that the smaller versions of these standard datasets were used. However this was complicated by the two major reasons:

* The smaller versions of most of the standard datsets were not available publically
* The ones that could be found (Oxford 17 categories dataset, Birds 200 categories) were getting corrupted by the system such that the dataloaders in PyTorch were reading in files that were prepended with a _ sign.

The [Tiny-Imagenet][9] dataset was used and the 200 odd classses were split into 4 tasks with 50 classes being assigned to each task randomly. This division can also be arbitrary and no special consideration has been given to the decision to split the dataset evenly. Each of these tasks has a "train" and a "test" folder to validate the performance on these wide ranging tasks.

Execute the following lines of code to download the Tiny-Imagenet dataset and split it into 4 folders belonging to different tasks

```sh
python3 data_prep.py
```

Description of the files in this repository
---------------------------------------------------

1) ``main.py``: Execute this file to train the model on the sequence of tasks
2) ``mas.py``: Contains functions that help in training and evaluating the model on these tasks (the forgetting <				that is undergone by the model)
3) ``model_class.py``: Contains the classes defining the model
4) ``model_train.py``: Contains the function that trains the model
5) ``optimizer_lib.py``: This file contains the optimizer classes, that realize the idea of computing the 								 gradients of the penalty term of the loss function locally 
6) ``data_prep.py``: File to download the datset and split the dataset into 4 folders that are interpreted as 						 different tasks 
7) ``utils/model_utils.py``: Utilities for training the model on the sequence of tasks
8) ``utils/mas_utils.py``: Utilities for the optimizers that implement the idea of computing the gradients       							locally

Training
------------------------------

To begin the training process on the sequence of tasks, use the **`main.py`** file. Simply execute the following lines to begin the training process

```sh
python3 main.py
```

The file takes the following arguments

* ***use_gpu***: Set the flag to true to train the model on the GPU **Default**: False
* ***batch_size***: Batch Size. **Default**: 8
* ***num_freeze_layers***: The number of layers in the feature extractor (features) of an Alexnet model, that you want to train. The rest are frozen and they are not trained. **Default**: 2
* ***num_epochs***: Number of epochs you want to train the model for. **Default**: 10
* ***init_lr***: Initial learning rate for the model. The learning rate is decayed every 20th epoch.**Default**: 0.001 
* ***reg_lambda***: The regularization parameter that provides the trade-off between the cross entropy loss function and the penalty for changes to important weights. **Default**: 0.01

Once you invoke the **`main.py`** module with the appropriate arguments, the following things shall happen

When the model fininshes being trained on a task, the last classification layer of the model (referred to as a classification head) is stored in a folder that is created for that specific task. This model stores the class specific features that are not shared across tasks. This folder also contains two text files **`performance.txt`** and **`classes.txt`**. The former records the performances of the model on the test sets, which is then used to compute the forgetting undergone by the model when the model is tested on the same task at the end of the training sequence. The latter records the information regarding the number of classes that the model was exposed to whilst being trained on that particular task. The rest of the model (referred to as shared_features) will be stored in the common folder to all the models as **`shared_model.pth`**. The reg_params associated with this model will be stored as a pickled file named as **`reg_params.pickle`**.\


The directory structure at the end of the training procedure, would resemble the following tree:

```
models
├── reg_params.pickle
├── shared_model.pth
├── Task_1
│   ├── classes.txt
│   ├── head.pth
│   └── performance.txt
├── Task_2
│  
├── Task_3
│   
└── Task_4
```

``head.pth.tar`` is the model file


Evaluating the model
-------------------------------

The model is evaluated at the end of the training sequence

The "forgetting" that the model has undergone on previous tasks whilst being trained on a sequence of tasks is computed and returned on the terminal. The function *compute_forgetting* reads in the previous performance from the ``performance.txt`` file stored in the folder specific to a task and compares it to the present performance of the model on that task. 


Results
-------------------------------
This paper is tested out on the tasks detailed in this [section][13]. Please note that the number of classes in each task have been halved to reduce experimentation time and the results obtained have been reported for this setting. All the models have been trained with the default values for the arguments taken by the **`main.py`**
module. 

| Task Number    | Forgetting (in %)|
| :------------: | :----------:     | 
|       1        |       10.2       |    
|       2        |       7.6        |    
|       3        |       4.1        |    
|       4        |       0          |    





To-Do's for this project
-------------------------------------

-[ ] Split the [MNIST dataset][12] to create another sequence of tasks and train the model on this sequence in addition to the tasks created from the Tiny_Imagenet dataset\
-[ ] Implement the idea of local Hebbian method (referred to in the paper as "local" method) which has not been implemented in the repository open sourced by the authors


References
-------------------------------
1. **Rahaf Aljundi, Francesca Babiloni, Mohamed Elhoseiny, Marcus Rohrbach, Tinne Tuytelaars** _Memory Aware Synapses: Learning what (not) to forget_ ECCV 2018. [[arxiv][10]]
2. **James Kirkpatrick, Razvan Pascanu, Neil Rabinowitz, Joel Veness, Guillaume Desjardins, Andrei A. Rusu, Kieran Milan, John Quan, Tiago Ramalho, Agnieszka Grabska-Barwinska, Demis Hassabis, Claudia Clopath, Dharshan Kumaran, Raia Hadsell** _Overcoming catastrophic forgetting in neural networks_ ICCV 2017 [[arxiv][3]]
3. **Rahaf Aljundi, Min Lin, Baptiste Goujaud, Yoshua Bengio** _Gradient based sample selection for online continual learning_ NeurIPS 2019 [[arxiv][1]]
3. **D.Hebb** _The Organization of behviour_ [[Book][2]]
4. PyTorch Docs. [[http://pytorch.org/docs/master](http://pytorch.org/docs/master)]

This repository owes a huge credit to the authors of the original [implementation][8]. This code repository could only be built due to the help offered by countless people on Stack Overflow and PyTorch Discus blogs


License
-------

BSD

[1]: https://arxiv.org/abs/1903.08671
[2]: http://s-f-walker.org.uk/pubsebooks/pdfs/The_Organization_of_Behavior-Donald_O._Hebb.pdf
[3]: https://arxiv.org/pdf/1612.00796.pdf
[4]: http://pytorch.org/docs/master
[5]: http://www.vision.caltech.edu/visipedia/CUB-200.html
[6]: http://places2.csail.mit.edu/
[7]: http://www.robots.ox.ac.uk/~vgg/data/flowers/17/
[8]: https://github.com/rahafaljundi/MAS-Memory-Aware-Synapses
[9]: https://tiny-imagenet.herokuapp.com/
[10]: https://arxiv.org/abs/1711.09601
[11]: http://s-f-walker.org.uk/pubsebooks/pdfs/The_Organization_of_Behavior-Donald_O._Hebb.pdf
[12]: http://yann.lecun.com/exdb/mnist/
[13]: #someid


