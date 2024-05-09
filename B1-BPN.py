'''
Viva Questions:-

1) Backpropagation is used
2) sigmod function is used because it is simple
3) Diffrentiation is used (basics of 11th)
4) numpy only is used
5) XOR function is used
6) in XOR, like inputs is 0, unlike input is 1

learning rate is the hyper parameter
epoch - one complete pass

This is a multilayered perceptron
'''



import numpy as np

# Sigmoid activation function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Derivative of sigmoid function
def sigmoid_derivative(x):
    return x * (1 - x)

# Input dataset
X = np.array([[0, 0],
              [0, 1],
              [1, 0],
              [1, 1]])

# Output dataset
y = np.array([[0],
              [1],
              [1],
              [0]])

# Initialize weights and biases randomly
input_size = 2
hidden_size = 2
output_size = 1

hidden_weights = np.random.uniform(size=(input_size, hidden_size))
hidden_biases = np.random.uniform(size=(1, hidden_size))

output_weights = np.random.uniform(size=(hidden_size, output_size))
output_biases = np.random.uniform(size=(1, output_size))

# Learning rate
learning_rate = 0.1

# Train the neural network
epochs = 10000

# heart of the algorithm
for epoch in range(epochs):
    
    # Forward propagation
    hidden_layer_input = np.dot(X, hidden_weights) + hidden_biases
    hidden_layer_output = sigmoid(hidden_layer_input)
    
    output_layer_input = np.dot(hidden_layer_output, output_weights) + output_biases
    output = sigmoid(output_layer_input)
    
    # Backpropagation
    output_error = y - output
    # output - predicted output
    # y - true output

    output_delta = output_error * sigmoid_derivative(output)
    
    # T - transpose to align dimensions
    hidden_error = output_delta.dot(output_weights.T)
    hidden_delta = hidden_error * sigmoid_derivative(hidden_layer_output)
    
    # Update weights and biases
    output_weights += hidden_layer_output.T.dot(output_delta) * learning_rate
    output_biases += np.sum(output_delta, axis=0, keepdims=True) * learning_rate
    
    hidden_weights += X.T.dot(hidden_delta) * learning_rate
    hidden_biases += np.sum(hidden_delta, axis=0, keepdims=True) * learning_rate

# Test the trained neural network
# Same as forward propagation
hidden_layer_input = np.dot(X, hidden_weights) + hidden_biases
hidden_layer_output = sigmoid(hidden_layer_input)
output_layer_input = np.dot(hidden_layer_output, output_weights) + output_biases
output = sigmoid(output_layer_input)

# Print the final output
print("Final Output:")
print(np.round(output))

