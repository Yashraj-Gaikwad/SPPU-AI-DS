'''
Viva Questions:-



'''


import numpy as np
import matplotlib.pyplot as plt

class Perceptron:
    def __init__(self, learning_rate=0.1, num_epochs=100):
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs

    def train(self, X, y):
        self.weights = np.zeros(X.shape[1] + 1)  # Additional weight for bias
        self.errors = []

        for _ in range(self.num_epochs):
            total_error = 0
            for xi, target in zip(X, y):
                update = self.learning_rate * (target - self.predict(xi))
                self.weights[1:] += update * xi
                self.weights[0] += update  # Update bias weight
                total_error += int(update != 0.0)
            self.errors.append(total_error)
            if total_error == 0:
                break

    def predict(self, X):
        activation = np.dot(X, self.weights[1:]) + self.weights[0]  # Include bias weight
        return np.where(activation >= 0, 1, -1)

# Example data
X = np.array([[2, 3], [4, 5], [7, 8], [9, 1]])
y = np.array([1, 1, -1, -1])

# Train the perceptron
perceptron = Perceptron()
perceptron.train(X, y)

# Plotting the decision boundary
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))
Z = perceptron.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.contourf(xx, yy, Z, alpha=0.4)
plt.scatter(X[:, 0], X[:, 1], c=y, marker='o', edgecolor='k')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Perceptron Decision Regions')
plt.show()
