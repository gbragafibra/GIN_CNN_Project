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
				# here we are using cross correlation of 1D inputs
				self.output[i] += signal.correlate1d(self.input[j],
					self.kernels[i,j], "valid")

		return self.output 

	def backward(self, output_grad, learning_rate):

		kernels_grad = np.zeros(self.kernels_shape)
		input_grad = np.zeros(self.input_shape)

		# Bias grad is just output grad
		for i in range(self.depth):
			for j in range(self.input_depth):
				kernels_grad[i,j] = signal.correlate1d(self.input[j],
					output_grad[i], "valid")
				input_grad[j] += signal.convolve1d(output_grad[i],
					self.kernels[i,j], "full")

		self.kernels -= learning_rate * kernels_grad
		self.biases -= learning_rate * output_grad

		return input_grad

## Testing GIN layer ########

class GIN(Layer):
	def __init__(self, input_size):
		self.W = np.random.randn(input_size, input_size)

	def forward(self, H, A):
		self.H = H
		self.A = A
		
		H_output = np.dot(np.dot(self.A, self.H).T, self.W)
		A_output = self.A * (H_output + H_output.T)		

		return H_output, A_output


	def backward(self, output_grad, learning_rate):
		grad_H = np.dot((np.dot(output_grad, self.W.T)).T,
			self.A + self.A.T)

		self.W -= learning_rate * np.dot(self.H.T, output_grad)

		return grad_H






class GlobalMeanPooling(Layer):

	def __init__(self):

		pass

	def forward(self, H):
		self.H_shape = H.shape

		return np.mean(H, axis = 1)

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

	def forward(self, H, A):
		pool_size = len(H) // self.size
		pooled_H = [np.mean(H[i * pool_size: (i+1) * pool_size])
		for i in range(self.size)]
		
		#Initialize
		pooled_A = np.zeros((self.size, self.size))		
		
		for i in range(self.size):
			for j in range(self.size):
				sub_A = A[i * pool_size: (i+1) * pool_size,
				j * pool_size: (j+1) * pool_size]

				pooled_A[i,j] = np.mean(sub_A)

			return pooled_H, pooled_A
	
	def backward(self):
		"""
		Given that this is to be applied as a
		first layer, and there are no learnable
		parameters, we can leave this empty (?)
		"""
		pass




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
        	
        	# loss 
			err += loss(y, output)

        	# Do backward
			grad = loss_prime(y, output)
			for layer in reversed(network):
				grad = layer.backward(grad, learning_rate)
        #Average loss
		err /= len(x_train)

		if verbose:
			print(f"{e + 1}/{epochs}, Loss = {err}")