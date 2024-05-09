'''
Viva Questions:-

Explain McCullochPitts
Explain ANDNOT

Also called as Linear threshold gate model

Neurons that wire together fire together
'''


import numpy as np

class McCullochPittsNeuron:
    def __init__(self, weights, threshold):
        self.weights    = weights
        self.threshold  = threshold

    def activate(self, inputs):
        # Compute the weighted sum of inputs
        weighted_sum = np.dot(inputs, self.weights)
        
        # Apply threshold logic
        if np.all(weighted_sum >= self.threshold):
            return 1
           
        else:
            return 0
            
def simulate_ANDNOT(inputs):
    
    # Define weights and threshold
    weights = np.array([-1, -1])
    threshold = -0.5
        
    # Create McCullochPitts Neuron
    andnot_neuron = McCullochPittsNeuron(threshold, weights)
        
    return andnot_neuron.activate(inputs)
        
        
# Test the ANDNOT function
print("ANDNOT(0, 0) = ", simulate_ANDNOT([0, 0]))
print("ANDNOT(0, 1) = ", simulate_ANDNOT([0, 1]))
print("ANDNOT(1, 0) = ", simulate_ANDNOT([1, 0]))
print("ANDNOT(1, 1) = ", simulate_ANDNOT([1, 1]))      