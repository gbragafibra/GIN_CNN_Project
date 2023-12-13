import numpy as np 
import random
from scipy import signal
from sklearn.model_selection import train_test_split
from rdkit import Chem
import pandas as pd
from preprocess_dataset import *
import matplotlib.pyplot as plt

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
		#Need self.activation_prime(self.input)[0]
		# as backprop in Sigmoid() in GIN layers
		# have two inputs H and A (tuple)
		# and we only want to consider H
		return np.multiply(output_grad, self.activation_prime(self.input)[0])


# Dense Layer (or Fully Connected Layer)

class Dense(Layer):

	def __init__(self, input_size, output_size, clip = 0.5):
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
		self.weights = np.random.randn(output_size, input_size)/np.sqrt(input_size)
		self.bias = np.random.randn(output_size, 1)
		self.clip = clip

	def forward(self, input):
		self.input = input.reshape(-1, 1)

		return np.dot(self.weights, self.input) + self.bias	

	def backward(self, output_grad, learning_rate):
		weights_grad = np.dot(output_grad, self.input.T)
		input_grad = np.dot(self.weights.T, output_grad)
		
		#Trying gradient clipping
		weights_grad = np.clip(weights_grad, -self.clip, self.clip)
		input_grad = np.clip(input_grad, -self.clip, self.clip)
		

		# Now update weights and biases
		self.weights -= learning_rate * weights_grad
		self.bias -= learning_rate * output_grad
		return input_grad




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
"""
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
"""
#Third time is the charm
"""
class Convolution(Layer):
    def __init__(self, input_length, kernel_size, depth):
        self.depth = depth
        self.input_length = input_length
        self.output_length = input_length - kernel_size + 1
        self.kernel_size = kernel_size
        self.kernels_shape = (depth, kernel_size)
        self.kernels = np.random.randn(*self.kernels_shape)
        self.biases = np.random.randn(depth, self.output_length)

    def forward(self, input):
    	self.input = input
    	self.output = np.copy(self.biases)
    	for i in range(self.depth):
    		for j in range(self.output_length):
    			self.output[i, j] += np.sum(self.input[j:j+self.kernel_size] * self.kernels[i])
    	return self.output

    def backward(self, output_grad, learning_rate):
        kernels_grad = np.zeros(self.kernels_shape)
        input_grad = np.zeros(self.input_length)

        for i in range(self.depth):
            for j in range(self.output_length):
                kernels_grad[i] += output_grad[i, j] * self.input[j:j+self.kernel_size]
                input_grad[j:j+self.kernel_size] += output_grad[i, j] * self.kernels[i]

        self.kernels -= learning_rate * kernels_grad
        self.biases -= learning_rate * output_grad
        return input_grad
"""
#Maybe fourth
class Convolution(Layer):
    def __init__(self, input_length, kernel_size, clip = 0.5):
        self.input_length = input_length
        self.output_length = input_length - kernel_size + 1
        self.kernel_size = kernel_size
        self.kernel = np.random.randn(kernel_size)/np.sqrt(input_length)
        self.biases = np.random.randn(self.output_length)
        self.clip = clip

    def forward(self, input):
        self.input = input
        self.output = np.copy(self.biases)
        for i in range(self.output_length):
            self.output[i] += np.sum(self.input[i:i+self.kernel_size] * self.kernel)
        return self.output

    def backward(self, output_grad, learning_rate):
        kernel_grad = np.zeros(self.kernel.shape)
        input_grad = np.zeros(self.input.shape)
        for i in range(self.output_length):
            for k in range(self.kernel_size):
                kernel_grad[k] += np.sum(output_grad[i] * self.input[i + k])
                """
                Need to flip kernel, so as to match
                a convolution op; Ideally it would
                be by 180 deg, but since it's 1D,
                we flip only horizontally
                """
                input_grad[i + k] += np.sum(output_grad[i] * np.flip(self.kernel[k]))
        
        if output_grad.ndim > 1:
        	output_grad = np.mean(output_grad, axis=1)

        kernel_grad = np.clip(kernel_grad, -self.clip, self.clip)
        input_grad = np.clip(input_grad, -self.clip, self.clip)

        self.kernel -= learning_rate * kernel_grad
        self.biases -= learning_rate * output_grad

        return input_grad


## Testing GIN layer ########

class GIN(Layer):
	def __init__(self, input_size, clip = 0.5):
		self.W = np.random.randn(input_size, input_size)/np.sqrt(input_size)
		self.clip = clip

	def forward(self, HandA):
		self.H, self.A = HandA
		self.H = np.array(self.H)
		self.A = np.array(self.A)
		
		
		H_output = np.dot(np.dot(self.A, self.H).T, self.W)
		A_output = self.A * (H_output + H_output.T)		

		return H_output, A_output


	def backward(self, output_grad, learning_rate):
		#grad_H = np.dot((np.dot(output_grad, self.W.T)),
		#	self.A + self.A.T)
		if output_grad.ndim == 3:
			output_grad_reshaped = output_grad.squeeze().T
		else:
			output_grad_reshaped = output_grad

		output_grad_reshaped = np.clip(output_grad_reshaped, -self.clip, self.clip)
		grad_H = np.dot(output_grad, np.dot(self.W.T, self.A))
		grad_H = np.clip(grad_H, -self.clip, self.clip)

		self.W -= learning_rate * np.dot(output_grad_reshaped, self.H.T)

		return grad_H






class GlobalMeanPooling(Layer):

	def __init__(self, clip = 0.5):

		self.clip = clip

	def forward(self, HandA):
		# Takes A, but doesn't use it
		# Given that the last GIN layer
		# has forward outputs of H and A
		H, A = HandA
		self.H_shape = H.shape
		if len(H.shape) == 1:
			H = H.reshape(-1, 1)
		#Testing here, mean over axis = 1
		pooled_H = np.mean(H, axis=1, keepdims=True)
		return pooled_H.T

	def backward(self, output_grad, learning_rate):
		#grad_H = np.tile(output_grad[:, np.newaxis],
		#	(1, self.H_shape[0])) / self.H_shape[0]
		grad_H = output_grad / self.H_shape[0]
		grad_H = np.clip(grad_H, -self.clip, self.clip)

		return grad_H
		


##########

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
	
	def backward(self, output_grad, learning_rate):
		"""
		Given that this is to be applied as a
		first layer, and there are no learnable
		parameters, we can leave this empty (?)
		"""
		pass

# Sum Pooling layer
class Sum_pooling(Layer):

	def __init__(self, size):
		self.size = size
	

	def forward(self, features):
		H, A = features
		
		pool_size = len(H) // self.size
		remainder = len(H) % self.size
		#List of strides
		strides = [pool_size + 1 if i < remainder else pool_size for i in range(self.size)]
		
		pooled_H = [np.sum(H[i * strides[i]: (i + 1) * strides[i]])
		for i in range(self.size)]
		
		#Initialize
		pooled_A = np.zeros((self.size, self.size))		
		
		for i in range(self.size):
			for j in range(self.size):
				sub_A = A[i * strides[i]: (i + 1) * strides[i],
				j * strides[j]: (j + 1) * strides[j]]

				pooled_A[i,j] = np.sum(sub_A)

		#Normalizing H and A
		pooled_H = (pooled_H - np.min(pooled_H))/(np.max(pooled_H) - np.min(pooled_H))
		pooled_A = (pooled_A - np.min(pooled_A))/(np.max(pooled_A) - np.min(pooled_A))

		return pooled_H, pooled_A
	
	def backward(self, output_grad, learning_rate):
		"""
		Given that this is to be applied as a
		first layer, and there are no learnable
		parameters, we can leave this empty (?)
		"""
		pass



# Batch Normalization Layer
class BatchNorm(Layer):
	def __init__(self, eps = 1e-5, momentum = 0.8, batch_size = 10, clip = 0.5):
		self.eps = eps 
		self.momentum = momentum
		self.running_mean = None
		self.running_var = None
		self.mean = None
		self.var = None
		self.gamma = None
		self.H_normalized = None
		self.batch_size = batch_size
		self.clip = clip

	def forward(self, H, Training = True):
		if self.running_mean is None:
			# Only for the first forward pass
			self.H = H
			self.running_mean = np.mean(H)
			self.running_var = np.var(H)
			self.gamma = 0.8
			self.beta = 0.4

		if Training:
			# All for the current batch
			self.H = H
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


		var_grad = np.sum(grad_H_normalized * (self.H - self.mean) * -0.5 * (self.var + self.eps)** (-3/2),
			axis = 0, keepdims = True)
		
		std_inv = 1/np.sqrt(self.var + self.eps)
		
		H_diff = 2*(self.H-self.mean)/self.batch_size

		mean_grad = np.sum(grad_H_normalized * (-std_inv), 
			axis = 0, keepdims = True) + var_grad * np.sum(-H_diff,
			axis = 0, keepdims = True)
		
		grad_H = grad_H_normalized * std_inv + var_grad * H_diff + mean_grad/self.batch_size

		""" #Not sure if both are equivalent
		grad_H = (1 / (self.batch_size * np.sqrt(self.var + self.eps)))*(self.batch_size * grad_H_normalized - 
				np.sum(grad_H_normalized, axis = 0) -
				self.H_normalized * np.sum(grad_H_normalized * self.H_normalized, axis = 0))

		"""
		grad_H = np.clip(grad_H, -self.clip, self.clip)
		

		self.gamma -= learning_rate * grad_gamma
		self.beta -= learning_rate * grad_beta

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
            if isinstance(x, tuple):  # For tuples
                return tuple(np.maximum(0, elem) for elem in x)
            else:  # A single array
                return np.maximum(0, x)

        def relu_prime(x):
            if isinstance(x, tuple):  # For tuples
                return tuple(np.where(elem > 0, 1, 0) for elem in x)
            else:  # For array
                return np.where(x > 0, 1, 0)

        super().__init__(relu, relu_prime)


def binary_cross_entropy(y_true, y_pred):
	return -np.mean(np.nan_to_num(y_true * np.log(y_pred) + (1 - y_true)*np.log(1 - y_pred)))

def binary_cross_entropy_prime(y_true, y_pred):
	return ((1 - y_true)/(1 - y_pred) - y_true/y_pred) / np.size(y_true)


#Mean squared error
def mse(y_true, y_pred):
	return np.mean((y_true - y_pred)**2)

def mse_prime(y_true, y_pred):
	return 2 * (y_pred - y_true)/np.size(y_true)

# Changed to accomodate BatchNorm training boolean
def predict(network, input, training = True):
	output = input 
	for layer in network:
		if isinstance(layer, BatchNorm):
			output = layer.forward(output, Training = training)

		else:
			output = layer.forward(output)

	return output

#Mini-batch Gradient Descent
def MBGD(network, loss, loss_prime, dataset, task, epochs = 10,
       learning_rate = 0.1, batch_size = 10, verbose = True):

    data_processed = preprocess_dataset(dataset, task) 
    
    features = [(entry["features"]["H"], entry["features"]["A"]) for entry in data_processed]
    labels = [entry["label"] for entry in data_processed]

    train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size=0.3)
    test_features, eval_features, test_labels, eval_labels = train_test_split(test_features, test_labels, test_size=0.5)

    train_losses = []
    for e in range(epochs):
        train_loss = 0
        train_correct = 0
        for i in range(0, len(train_features), batch_size):
            x_batch = train_features[i: i + batch_size]
            y_batch = train_labels[i: i + batch_size]

            batch_loss = 0
            batch_grad = 0 
            for x, y in zip(x_batch, y_batch):
                # Do forward pass
                output = predict(network, x, training = True)
                batch_loss += loss(y, output)

                if (output > 0.5 and y == 1) or (output < 0.5 and y == 0):
                	train_correct += 1

                
                batch_grad += loss_prime(y, output)
            # Do backward pass
            batch_grad /= len(x_batch)
            for layer in reversed(network):
                batch_grad = layer.backward(batch_grad, learning_rate)
        
            train_loss += batch_loss

        train_loss /= len(train_features)
        train_accuracy = train_correct/len(train_features)
        train_losses.append(train_loss)
        if verbose:
            print(f"Epoch {e + 1}/{epochs}, Training Loss = {train_loss}, Training accuracy = {train_accuracy}")


    plt.plot(np.arange(1,epochs + 1,1), train_losses, "k")
    plt.xlabel("Epoch")
    plt.ylabel("Training Loss")

    test_loss = 0
    test_correct = 0
    for x, y in zip(test_features, test_labels):
        output = predict(network, x, training = False)
        test_loss += loss(y, output)
        if (output > 0.5 and y == 1) or (output < 0.5 and y == 0):
        	test_correct += 1
    test_loss /= len(test_features)
    test_accuracy = test_correct/len(test_features)
    print(f"Test Loss: {test_loss}, Test accuracy: {test_accuracy}")

    eval_loss = 0
    eval_correct = 0
    for x, y in zip(eval_features, eval_labels):
        output = predict(network, x, training = False)
        eval_loss += loss(y, output)
        if (output > 0.5 and y == 1) or (output < 0.5 and y == 0):
        	eval_correct += 1   
    eval_loss /= len(eval_features)
    eval_accuracy = eval_correct/len(eval_features)
    print(f"Evaluation Loss: {eval_loss}, Evaluation accuracy: {eval_accuracy}")