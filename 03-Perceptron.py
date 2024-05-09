'''
Viva Questions:-

Explain Perceptron

1) multilayer neural network

2) Linearly Separable



'''


import numpy as np

class Perceptron:
    def __init__(self, input_size, learning_rate=0.1):
    
        self.weights = np.zeros(input_size)
        self.learning_rate = learning_rate
    
    def predict(self, inputs):
        summation = np.dot(inputs, self.weights)
        
        return 1 if summation > 0 else 0
    
    def train(self, inputs, target):
    
        inputs = np.array(inputs)  # Convert inputs to NumPy array
        prediction = self.predict(inputs)
        error = target - prediction
        self.weights += self.learning_rate * error * inputs

# Function to convert decimal number to binary representation
def decimal_to_binary(decimal_num, num_bits):

    binary = bin(decimal_num)[2:]
    binary = '0' * (num_bits - len(binary)) + binary
    
    return [int(bit) for bit in binary]

# Train perceptron on dataset
def train_perceptron(dataset, perceptron, num_epochs):
    for _ in range(num_epochs):
        for input_data, label in dataset:
            perceptron.train(input_data, label)

# Main function

num_bits = 4  # Number of bits for binary representation
max_num = 15  # Maximum number to consider
num_epochs = 1000  # Number of training epochs

# Generate dataset
dataset = [(decimal_to_binary(i, num_bits), 1 if i % 2 == 0 else 0) for i in range(1, max_num + 1)]

# Initialize and train perceptron
perceptron = Perceptron(input_size=num_bits)
train_perceptron(dataset, perceptron, num_epochs)

# Take input from user
input_num = int(input("Enter a number: "))
input_binary = decimal_to_binary(input_num, num_bits)

# Predict whether the input number is even or odd
prediction = perceptron.predict(input_binary)

if prediction == 1:
    print(f'The number {input_num} is even.')
else:
    print(f'The number {input_num} is odd.')
    
    
