import numpy as np 
import random
from scipy import signal

# General layer class
class Layer:

	def __init__(self):
		self.input = None
		self.output = None

	def forward(self, input):

		pass


	def backward(self, output_grad, learning_rate):

		pass

# General activation Layer

class Activation(Layer):

	def __init__(self, activation, activation_prime):
		self.activation = activation
		self.activation_prime = activation_prime

	def forward(self, input):
		self.input = input 

		return self.activation(self.input)

	def backward(self, output_grad, learning_rate):

		return np.multiply(output_grad, self.activation_prime(self.input))


# Dense Layer (or Fully Connected Layer)

class Dense(Layer):

	def __init__(self, input_size, output_size):
		"""
		Initialize weights and biases
		We are sampling from standard normal dist
		Might possibly want to normalize 
		by sqrt(n) so as to not have
		slow learning of params
		if using activation functions
		with low derivatives for
		"a" close to 1 or 0
		"""
		self.weights = np.random.randn(output_size, input_size)
		self.bias = np.random.randn(output_size, 1)
		

	def forward(self, input):
		self.input = input 

		return np.dot(self.weights, self.input) + self.bias	

	def backward(self, output_grad, learning_rate):
		weights_grad = np.dot(output_grad, self.input.T)
		input_grad = np.dot(self.weights.T, output_grad)
		# Now update weights and biases
		self.weights -= learning_rate * weights_grad
		self.bias -= learning_rate * output_grad

		return np.dot(self.weights.T, output_grad)


# Convolutional layer 
class Convolution(Layer):

	def __init__(self, input_shape, kernel_size, depth):
		# depth -> how many kernels do we want
		input_depth, input_height, input_width = input_shape
		self.depth = depth
		self.input_shape = input_shape
		self.input_depth = input_depth
		self.output_shape = (depth, input_height - kernel_size + 1,
			input_width - kernel_size + 1)
		self.kernels_shape = (depth, input_depth, kernel_size, kernel_size)			
		# Initialize kernels params
		self.kernels = np.random.randn(*self.kernels_shape)
		self.biases = np.random.randn(*self.output_shape)
		# * before self, unpacks the tuple
		# that we are expecting

	def forward(self, input):
		self.input = input 
		# here we can have the outputs
		# with the bias values already
		self.output = np.copy(self.biases)
		for i in range(self.depth):
			for j in range(self.input_depth):
				# here we are using cross correlation of 2D inputs
				self.output[i] += signal.correlate2d(self.input[j],
					self.kernels[i,j], "valid")

		return self.output 

	def backward(self, output_grad, learning_rate):

		kernels_grad = np.zeros(self.kernels_shape)
		input_grad = np.zeros(self.input_shape)

		# Bias grad is just output grad
		for i in range(self.depth):
			for j in range(self.input_depth):
				kernels_grad[i,j] = signal.correlate2d(self.input[j],
					output_grad[i], "valid")
				input_grad[j] += signal.convolve2d(output_grad[i],
					self.kernels[i,j], "full")

		self.kernels -= learning_rate * kernels_grad
		self.biases -= learning_rate * output_grad

		return input_grad

## Testing GIN layer ########

class GIN(Layer):
	def __init__(self):

		pass









##########


class Sigmoid(Activation):
	def __init__(self):
		def sigmoid(x):
			return 1/(1 + np.exp(-x))

		def sigmoid_prime(x):
			sig = sigmoid(x)
			return sig * (1 - sig)

		super().__init__(sigmoid, sigmoid_prime)


class ReLU(Activation):
	def __init__(self):
		def relu(x):
			return np.maximum(0,x)

		def relu_prime(x):
			return np.where(x > 0, 1, 0)

		super().__init__(relu, relu_prime)

def binary_cross_entropy(y_true, y_pred):
	return -np.mean(np.nan_to_num(y_true * np.log(y_pred) + (1 - y_true)*np.log(1 - y_pred)))

def binary_cross_entropy_prime(y_true, y_pred):
	return ((1 - y_true)/(1 - y_pred) - y_true/y_pred) / np.size(y_true)


def predict(network, input):
	output = input 
	for layer in network:
		output = layer.forward(output)

	return output

def train(network, loss, loss_prime, x_train, y_train,
	epochs = 10, learning_rate = 0.1, verbose = True):

	for e in range(epochs):
		err = 0
		for x,y in zip(x_train,y_train):
			# Do forward
			output = predict(network, x)
        	
        	# error (Useless, just to have an idea
        	# of it on screen)
			err += loss(y, output)

        	# Do backward
			grad = loss_prime(y, output)
			for layer in reversed(network):
				grad = layer.backward(grad, learning_rate)
        #Average error
		err /= len(x_train)

		if verbose:
			print(f"{e + 1}/{epochs}, error = {err}")