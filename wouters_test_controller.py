import sys, os

from evoman.controller import Controller
import numpy as np
from typing import Optional

# imports other libs
import numpy as np
from nnlayers import *

# implements controller structure for player
class player_controller(Controller):
	def __init__(self, _n_hidden: list[int]):
		self.n_hidden = _n_hidden

	def set(self, layerconf: Optional[list[tuple[np.ndarray,np.ndarray]]], n_inputs: int):
		# Number of hidden neurons
			
		# We must assure that the amount of layers we have weight and bias data for matches the amount of layers we want to create
		assert len(layerconf) == len(self.n_hidden), "The amount of configurations for the hidden layers does not match the amount of hidden layers"
		
		self.layers = []
		in_size = n_inputs
		for i, layer_size in enumerate(self.n_hidden):
			# print(layer_size)
			# print(layerconf)
			self.layers.append(Linear(in_size, layer_size, layerconf[i][0], layerconf[i][1]))
            

	def control(self, inputs, controller):
		# Normalises the input using min-max scaling
		inputs = (inputs-min(inputs))/float((max(inputs)-min(inputs)))

		# Preparing the weights and biases from the controller of layer 1
		for layer in self.layers[:-1]:
			output = sigmoid_activation(layer.forward(inputs))
			inputs = output
		
		output = sigmoid_activation(self.layers[-1].forward(inputs))[0]

		# print(output.shape)
		# takes decisions about sprite actions
		if output[0] > 0.5:
			left = 1
		else:
			left = 0

		if output[1] > 0.5:
			right = 1
		else:
			right = 0

		if output[2] > 0.5:
			jump = 1
		else:
			jump = 0

		if output[3] > 0.5:
			shoot = 1
		else:
			shoot = 0

		if output[4] > 0.5:
			release = 1
		else:
			release = 0

		return [left, right, jump, shoot, release]


# implements controller structure for enemy
class enemy_controller(Controller):
	def __init__(self, _n_hidden):
		# Number of hidden neurons
		self.n_hidden = [_n_hidden]

	def control(self, inputs,controller):
		# Normalises the input using min-max scaling
		inputs = (inputs-min(inputs))/float((max(inputs)-min(inputs)))

		if self.n_hidden[0]>0:
			# Preparing the weights and biases from the controller of layer 1

			# Biases for the n hidden neurons
			bias1 = controller[:self.n_hidden[0]].reshape(1,self.n_hidden[0])
			# Weights for the connections from the inputs to the hidden nodes
			weights1_slice = len(inputs)*self.n_hidden[0] + self.n_hidden[0]
			weights1 = controller[self.n_hidden[0]:weights1_slice].reshape((len(inputs),self.n_hidden[0]))

			# Outputs activation first layer.
			output1 = leakyrelu_activation(inputs.dot(weights1) + bias1)

			# Preparing the weights and biases from the controller of layer 2
			bias2 = controller[weights1_slice:weights1_slice + 5].reshape(1,5)
			weights2 = controller[weights1_slice + 5:].reshape((self.n_hidden[0],5))

			# Outputting activated second layer. Each entry in the output is an action
			output = sigmoid_activation(output1.dot(weights2)+ bias2)[0]
		else:
			bias = controller[:5].reshape(1, 5)
			weights = controller[5:].reshape((len(inputs), 5))

			output = sigmoid_activation(inputs.dot(weights) + bias)[0]

		# takes decisions about sprite actions
		if output[0] > 0.5:
			attack1 = 1
		else:
			attack1 = 0

		if output[1] > 0.5:
			attack2 = 1
		else:
			attack2 = 0

		if output[2] > 0.5:
			attack3 = 1
		else:
			attack3 = 0

		if output[3] > 0.5:
			attack4 = 1
		else:
			attack4 = 0

		return [attack1, attack2, attack3, attack4]
