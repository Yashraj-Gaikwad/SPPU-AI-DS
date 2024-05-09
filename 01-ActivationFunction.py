'''
Viva Questions:-

use of numpy
use of matplotlib

'''

import numpy as np
import matplotlib.pyplot as plt

# Define Activation  

# Sigmoid 
# Non Linear activation function, 0 - 1, smooth, squashing function, non zero centered
# Activation Threshold at 0.5
# Used - output layer
def sigmoid(x):
    return 1/(1+ np.exp(-x))
    
# Rectifed Linear Unit
# Most commonly used activation function
# Linear function, Sparse inducing property, does not suffer from vanishing gradient, Efficent
# Limitation = dying RELU
# Variant = Leaky RElU
def relu(x):
    return np.maximum(0, x)

# Hyperbolic Tangent
# Formula = (e^x - e^-x)/(e^x + e^-x)
# Range = -1 to 1
# soomth, zero centered, squashing function
# Activation Threshold at 0
# Usage = RNNs, LSTMs, Hidden Layer

def tanh(x):
    return np.tanh(x)
    
# Softmax
# Non-Linear, output distribution, normalization, Cross entropy Loss, Differentiable
# Used in last layer
# Based on probability

def softmax(x):
    # Subtracting the maximum value for numerical stability
    exp_x = np.exp(x - np.max(x))
    
    return exp_x / exp_x.sum(axis=0)
    
# Define input range
x = np.linspace(-5, 5, 100)

# Plot activation functions
plt.figure(figsize=(12, 8))

plt.subplot(2, 2, 1)
plt.plot(x, sigmoid(x), label='Sigmoid')
plt.title('Sigmoid Activation Function')
plt.xlabel('Input')
plt.ylabel('Output')
plt.legend()


plt.subplot(2, 2, 2)
plt.plot(x, relu(x), label = 'ReLU')
plt.title('ReLU Activation Function')
plt.xlabel('Input')
plt.ylabel('Output')
plt.legend()


plt.subplot(2, 2, 3)
plt.plot(x, tanh(x), label = 'Tanh')
plt.title('Tanh Activation Function')
plt.xlabel('Input')
plt.ylabel('Output')
plt.legend()


plt.subplot(2, 2, 4)
plt.plot(x, softmax(x), label = 'Softmax')
plt.title('Softmax Activation Function')
plt.xlabel('Input')
plt.ylabel("Output")
plt.legend()


plt.tight_layout()

plt.savefig('Sigmoid_Activation_Functions.png')

plt.show()






