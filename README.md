# avalam_self_play
Avalam - Self Play Agent

This repository contains the source code for an AI agent designed to play the board game Avalam.
The agent uses Monte Carlo Tree Search (MCTS) and a neural network for decision making.

Repository Structure :
The repository is organized by grid size, with each grid size having its own directory:

	mini_Avalam_3x3/: Contains the code for training and testing the AI agent on a 3x3 grid size of Avalam.
	mini_Avalam_5x5/: Contains the code for training and testing the AI agent on a 5x5 grid size of Avalam.
	Avalam_9x9/: Contains the code for training and testing the AI agent on the original 9x9 grid size of Avalam.

Within each directory, you'll find the following files:

	mini_Avalam_{size}.ipynb: The main notebook for running the AI agent.
	self_play_mini_Avalam_{size}.py: Contains the ResNet implementation for the AI agent.
	mini_Avalam_{size}.py: Contains the rules of the Avalam game, implemented as an array representation for self-play.
	data_augmentation.py: Contains the implementation of data augmentation technique as explained in the report.
	Model and optimizer files for different iterations and training methods. 
	For example, model_{n_iter}.pt is the model file after n_iter iterations of self-play training.
	Please note that size refers to the grid size, and n_iter refers to the number of training iterations.
	
The models are saved as follow : 

	self play : model_{n_iter}.pt
	self with parallelization : model_paral_{n_iter}.pt 
	model trained with supervised learning : model_supervised_{n_iter}.pt
	model trained with supervised learning and a bigger model : model_supervised_big_model_{n_iter}.pt 
	Main avalam 9x9, the main model that took 1 day of training is : model_supervised_big_model_final_0.pt
	

Getting Started:

	Clone the repository: git clone https://github.com/sabrikhalil/avalam_self_play
	Navigate into the avalam_self_play directory 
	Choose the grid size you're interested in and navigate into the corresponding directory: cd mini_Avalam_3x3
	Open the main notebook: jupyter notebook mini_Avalam_3x3.ipynb


Requirements :

The AI agent was developed using Python 3.7. The following libraries are required:
	
	PyTorch
	NumPy
	tqdm.notebook

Please ensure that you have the necessary hardware and computational resources for training the agent, especially for the 9x9 grid size.

