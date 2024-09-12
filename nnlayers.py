import numpy as np
from typing import Optional

def sigmoid_activation(x: np.ndarray):
	return 1./(1.+np.exp(-x))

def leakyrelu_activation(x: np.ndarray):
	return np.maximum(0, x) + np.minimum(0, 0.01*x)

def normalise(x: np.ndarray):
    return (x-np.min(x))/float((np.max(x)-np.min(x)))

class Linear:
    def __init__(self, input_size: int, output_size: int, weights: np.ndarray, bias: np.ndarray):
        """Initialises Linear layer with weights and bias.

        Args:
            input_size (int): The amount of input values this layer will receive.
            output_size (int): The amount of values this layer will output. (Aka the amount of neurons in this layer)
            weights (np.ndarray): Weights that are used in this layer, shape should be (input_size, output_size)
            bias (np.ndarray): Bias values for this layer, shape should be (1, output_size)
        """
        self.input_size = input_size
        self.output_size = output_size

        self.weights = weights
        self.bias = bias
        
    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass for this layer.

        Args:
            x (np.ndarray): Input values for this layer.

        Returns:
            np.ndarray: Output values for this layer.
        """
        return x.dot(self.weights) + self.bias