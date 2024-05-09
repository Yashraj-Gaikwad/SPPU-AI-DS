'''
Viva Questions:-

Explain BAM

'''



import numpy as np

class BAM:
    def __init__(self, input_size, output_size):
        self.input_size = input_size
        self.output_size = output_size
        self.W = np.zeros((input_size, output_size))
        self.T = np.zeros((output_size, input_size))

    def train(self, X, Y):
        for x, y in zip(X, Y):
            self.W += np.outer(x, y)
            self.T += np.outer(y, x)

    def recall(self, X, max_iterations=100):
        recalled = []
        for x in X:
            prev_output = np.zeros(self.output_size)
            output = self.compute_output(x, prev_output, max_iterations)
            recalled.append(output)
        return np.array(recalled)

    def compute_output(self, x, prev_output, max_iterations):
        for _ in range(max_iterations):
            y = self.activate_output(x, prev_output)
            x = self.activate_input(y)
            if np.array_equal(x, prev_output):
                break
            prev_output = x
        return x

    def activate_output(self, x, prev_output):
        return np.dot(x, self.W)

    def activate_input(self, y):
        return np.dot(y, self.T)

# Example usage:
if __name__ == "__main__":
    # Define input and output vectors
    X = np.array([[1, 0], [0, 1], [1, 1]])
    Y = np.array([[1, 0, 0], [0, 1, 1], [1, 1, 0]])

    # Initialize BAM
    bam = BAM(input_size=2, output_size=3)

    # Train BAM
    bam.train(X, Y)

    # Test recall
    test_input = np.array([[1, 0], [0, 1], [1, 1]])
    recalled_outputs = bam.recall(test_input)
    print("Recalled outputs:")
    print(recalled_outputs)
