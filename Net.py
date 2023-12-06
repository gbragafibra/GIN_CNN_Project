import numpy as np 
import random
from scipy import signal
from sklearn.model_selection import train_test_split
from rdkit import Chem
import pandas as pd
from preprocess_dataset import *

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

		return np.dot(self.weights, self.input.T) + self.bias	

	def backward(self, output_grad, learning_rate):
		weights_grad = np.dot(output_grad, self.input)
		input_grad = np.dot(self.weights.T, output_grad)
		# Now update weights and biases
		self.weights -= learning_rate * weights_grad
		self.bias -= learning_rate * output_grad

		return np.dot(self.weights.T, output_grad)


# Convolutional layer 
"""
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
				# here we are using cross correlation of 1D inputs
				self.output[i] += signal.correlate(self.input[j],
					self.kernels[i,j], "valid")

		return self.output 

	def backward(self, output_grad, learning_rate):

		kernels_grad = np.zeros(self.kernels_shape)
		input_grad = np.zeros(self.input_shape)

		# Bias grad is just output grad
		for i in range(self.depth):
			for j in range(self.input_depth):
				kernels_grad[i,j] = signal.correlate(self.input[j],
					output_grad[i], "valid")
				input_grad[j] += signal.convolve(output_grad[i],
					self.kernels[i,j], "full")

		self.kernels -= learning_rate * kernels_grad
		self.biases -= learning_rate * output_grad

		return input_grad
"""
#2nd version; Focus on 1D inputs
class Convolution(Layer):
    def __init__(self, input_shape, kernel_size, depth):
        input_depth, input_length = input_shape
        self.depth = depth
        self.input_shape = input_shape
        self.input_depth = input_depth
        self.output_length = input_length - kernel_size + 1
        self.kernels_shape = (depth, input_depth, kernel_size)
        self.kernels = np.random.randn(*self.kernels_shape)
        self.biases = np.random.randn(depth, self.output_length)

    def forward(self, input):
        self.input = input
        self.output = np.copy(self.biases)
        for i in range(self.depth):
            for j in range(self.input_depth):
                self.output[i] += np.convolve(self.input[j], self.kernels[i, j], mode='valid')
        return self.output

    def backward(self, output_grad, learning_rate):
        kernels_grad = np.zeros(self.kernels_shape)
        input_grad = np.zeros(self.input_shape)

        for i in range(self.depth):
            for j in range(self.input_depth):
                kernels_grad[i, j] = np.correlate(self.input[j], output_grad[i], mode='valid')
                input_grad[j] += np.convolve(output_grad[i], np.flip(self.kernels[i, j]), mode='full')

        self.kernels -= learning_rate * kernels_grad
        output_grad = np.sum(output_grad, axis=0, keepdims=True)
        self.biases -= learning_rate * output_grad

        return input_grad


## Testing GIN layer ########

class GIN(Layer):
	def __init__(self, input_size):
		self.W = np.random.randn(input_size, input_size)

	def forward(self, HandA):
		H, A = HandA

		
		H_output = np.dot(np.dot(A, H).T, self.W)
		A_output = A * (H_output + H_output.T)		

		return H_output, A_output


	def backward(self, output_grad, learning_rate):
		grad_H = np.dot((np.dot(output_grad, self.W.T)).T,
			A + A.T)

		self.W -= learning_rate * np.dot(H.T, output_grad)

		return grad_H






class GlobalMeanPooling(Layer):

	def __init__(self):

		pass

	def forward(self, HandA):
		# Takes A, but doesn't use it
		# Given that the last GIN layer
		# has forward outputs of H and A
		H, A = HandA
		self.H_shape = H.shape
		#Testing here, mean over axis = 1
		return np.mean(H, axis=1)

	def backward(self, output_grad):
		grad_H = np.tile(output_grad[:, np.newaxis],
			(1, self.H_shape[1])) / self.H_shape[1]

		return grad_H
		


##########

# 1D Mean pooling layer
# Will probably need this
# to have fixed size inputs
# Both regarding node features
# vectors as edge features ones

#Perhaps need to change this
# into also taking the adjacency
# matrix. Such that we'll have mean
# pooling of H and Ã

# Afterall, this would be performing
# mean pooling both for A and H
# A being 2D case, and H 1D case
class Mean_pooling(Layer):

	def __init__(self, size):
		self.size = size
	

	def forward(self, features):
		H, A = features
		
		pool_size = len(H) // self.size
		remainder = len(H) % self.size
		#List of strides
		strides = [pool_size + 1 if i < remainder else pool_size for i in range(self.size)]
		
		pooled_H = [np.mean(H[i * strides[i]: (i + 1) * strides[i]])
		for i in range(self.size)]
		
		#Initialize
		pooled_A = np.zeros((self.size, self.size))		
		
		for i in range(self.size):
			for j in range(self.size):
				sub_A = A[i * strides[i]: (i + 1) * strides[i],
				j * strides[j]: (j + 1) * strides[j]]

				pooled_A[i,j] = np.mean(sub_A)

		return pooled_H, pooled_A
	
	def backward(self):
		"""
		Given that this is to be applied as a
		first layer, and there are no learnable
		parameters, we can leave this empty (?)
		"""
		pass


# Batch Normalization Layer
class BatchNorm(Layer):
	def __init__(self, eps = 1e-5, momentum = 0.8, batch_size = 10):
		self.eps = eps 
		self.momentum = momentum
		self.running_mean = None
		self.running_var = None
		self.mean = None
		self.var = None
		self.gamma = None
		self.H_normalized = None
		self.batch_size = batch_size

	def forward(self, H, Training = True):
		if self.running_mean is None:
			# Only for the first forward pass
			self.running_mean = np.mean(H)
			self.running_var = np.var(H)
			self.gamma = 1.0
			self.beta = 0.1

		if Training:
			# All for the current batch
			self.mean = np.mean(H)
			self.var = np.var(H)
			self.H_normalized = (H - self.mean) / np.sqrt(self.var + self.eps)

			H_out = self.gamma * self.H_normalized + self.beta

			#Updating running stats
			self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * self.mean 
			self.running_var = self.momentum * self.running_var + (1 - self.momentum) * self.var
		
		#For testing and evaluation
		else:
			self.H_normalized = (H - self.running_mean) / np.sqrt(self.running_var + self.eps)
			H_out = self.gamma * self.H_normalized + self.beta

		return H_out		
	
	def backward(self, output_grad, learning_rate):
		grad_gamma = np.sum(output_grad * self.H_normalized)
		grad_beta = np.sum(output_grad)

		grad_H_normalized = output_grad * self.gamma 

		grad_H = (1 / (self.batch_size * np.sqrt(self.var + self.eps)))*(self.batch_size * grad_H_normalized - 
				np.sum(grad_H_normalized, axis = 0) -
				self.H_normalized * np.sum(grad_H_normalized * self.H_normalized, axis = 0))


		self.gamma -= (learning_rate/self.batch_size) * grad_gamma
		self.beta -= (learning_rate/self.batch_size) * grad_beta

		return grad_H 

# Reshape Layer
class Reshape(Layer):

	def __init__(self, input_shape, output_shape):
		self.input_shape = input_shape
		self.output_shape = output_shape

	def forward(self, input):
		return np.reshape(input, self.output_shape)

	def backward(self, output_grad, learning_rate):
		return np.reshape(output_grad, self.input_shape)



class Sigmoid(Activation):
	def __init__(self):
		def sigmoid(x):
			if isinstance(x, tuple): #For tuples
				return tuple(1 / (1 + np.exp(-elem)) for elem in x)

			else: # A single array
				return 1/(1 + np.exp(-x))

		def sigmoid_prime(x):
			sig = sigmoid(x)
			if isinstance(x, tuple): #for tuples
				return tuple(sigmoid(elem) * (1 - sigmoid(elem)) for elem in x)
			
			else:#for array
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

def SGD(network, loss, loss_prime, dataset, task, epochs = 10,
       learning_rate = 0.1, batch_size = 10, verbose = True):

    data_processed = preprocess_dataset(dataset, task) 
    
    features = [(entry["features"]["H"], entry["features"]["A"]) for entry in data_processed]
    labels = [entry["label"] for entry in data_processed]

    train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size=0.2)
    test_features, eval_features, test_labels, eval_labels = train_test_split(test_features, test_labels, test_size=0.5)


    for e in range(epochs):
        train_loss = 0
        for i in range(0, len(train_features), batch_size):
            x_batch = train_features[i: i + batch_size]
            y_batch = train_labels[i: i + batch_size]

            batch_loss = 0
            for x, y in zip(x_batch, y_batch):
                # Do forward pass
                output = predict(network, x)
                batch_loss += loss(y, output) / len(x_batch)

                # Do backward pass
                batch_grad = loss_prime(y, output) / len(x_batch)
                for layer in reversed(network):
                    batch_grad = layer.backward(batch_grad, learning_rate)
            
            train_loss += batch_loss

        train_loss /= len(train_features)

        if verbose:
            print(f"Epoch {e + 1}/{epochs}, Training Loss = {train_loss}")


    test_loss = 0
    for x, y in zip(test_features, test_labels):
        output = predict(network, x)
        test_loss += loss(y, output)
    test_loss /= len(test_features)
    print(f"Test Loss: {test_loss}")

    eval_loss = 0
    for x, y in zip(eval_features, eval_labels):
        output = predict(network, x)
        eval_loss += loss(y, output)
    eval_loss /= len(eval_features)
    print(f"Evaluation Loss: {eval_loss}")