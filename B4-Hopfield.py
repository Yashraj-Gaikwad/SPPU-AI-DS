'''
Viva Questions:-

Explain Hopfield
Unsupervised Learning Algorithm


'''



import numpy as np

class HopfieldNetwork:
    def __init__(self, num_neurons):
        self.num_neurons = num_neurons
        self.weights = np.zeros((num_neurons, num_neurons))

    def train(self, patterns):
        num_patterns = len(patterns)
        
        for pattern in patterns:
            pattern_row = np.reshape(pattern, (1, self.num_neurons))
            self.weights += np.dot(pattern_row.T, pattern_row)
       
        # Sets diagonal elements to zero, which are self-connections
        np.fill_diagonal(self.weights, 0)
        
        # normalization
        self.weights /= num_patterns

    def recall(self, input_pattern, max_iter=100):
        output_pattern = np.copy(input_pattern)
        
        for _ in range(max_iter):
            new_pattern = np.sign(np.dot(output_pattern, self.weights))
            
            if np.array_equal(new_pattern, output_pattern):
                break
            
            output_pattern = new_pattern
        
        return output_pattern

# Define the patterns to store
patterns = [
    [1, -1, 1, -1],
    [-1, -1, 1, 1],
    [1, 1, -1, -1],
    [-1, 1, -1, 1]
]

# Initialize and train the Hopfield Network
num_neurons = len(patterns[0])
hopfield_net = HopfieldNetwork(num_neurons)
hopfield_net.train(patterns)

# Test the network by recalling stored patterns
for pattern in patterns:
    recalled_pattern = hopfield_net.recall(pattern)
    print("Recalled Pattern:", recalled_pattern)
