import numpy as np
import random

# Two input classes (A and B) each with 1 example in the "dataset". 
input_a = np.random.rand(16)
input_b = np.random.rand(16)

# --------------------------------------- Hyperparameters -----------------------------------------
n_hidden = 100
threshold = 1.3 * len(input_a)*0.5
learning_rate = 0.01
train_iters = 100
test_iters = 100
# Number of times a given input generates spikes in the input neurons before a new input is passed
time = 5
# -------------------------------------------------------------------------------------------------
def initiate_weights():
    n_input = len(input_a)
    # One row for each input neuron's weights
    weights = np.random.rand(n_input, n_hidden)
    return weights


def process_input_neurons(weight_matrix, inpt, time, train=True):
    hidden_activations = np.zeros(n_hidden)
    firings = np.zeros(n_hidden)
    
    for i in range(time):
        # We want to process each input, but in a random order
        indices = np.arange(0, len(input_a))
        np.random.shuffle(indices)
        
        # Calculate hidden activations
        for i in indices:
            input_neuron_weights = weight_matrix[i]
            for j in range(0, n_hidden):
                hidden_activations[j] += input_neuron_weights[j] * inpt[i]

                # Did the activation cause the neuron to fire?
                if hidden_activations[j] >= threshold:
                    hidden_activations[j] = 0
                    firings[j] += 1

                    # THIS IS THE HEBBIAN LEARNING BIT. WEIGHT IS INCREASED BETWEEN THE FIRING 
                    # NEURON AND THE NEURON THAT MADE IT FIRE!
                    if train:
                        weight_matrix[i][j] += (1 - weight_matrix[i][j]) * learning_rate


    return weight_matrix, firings
    
def assign_categories(firings_a, firings_b):
    prefferences = firings_a - firings_b
    for hidden_neuron in range(len(prefferences)):
        if prefferences[hidden_neuron] > 0: # If the neuron preferres input A
            prefferences[hidden_neuron] = 1
        else:                              # If the neuron preferres input B
            prefferences[hidden_neuron] = 0
    return prefferences

def train(iters, time, weights):
    firings_a = np.zeros(n_hidden)
    firings_b = np.zeros(n_hidden)
    assignments = np.zeros(n_hidden)

    for i in range(iters):

        # Pick an input image randomly and process it
        if bool(random.getrandbits(1)):
            inpt = input_a
            weights, firings = process_input_neurons(weights, inpt, time)
            firings_a += firings
        else:
            inpt = input_b
            weights, firings = process_input_neurons(weights, inpt, time)
            firings_b += firings
        
        # Print accuracy every 10 iterations
        if i % 10 == 0:
            assignments = assign_categories(firings_a, firings_b)
            print("Accuracy at iteration %2i: %0.2f" % (i, test(time, weights, assignments)))
            print("Average weight strength: %.3f" % np.mean(weights))

    return weights, assignments


def test(time, weights, assignments):
    right = 0
    wrong = 0

    for i in range(test_iters):
        
        # Pick an input image randomly and process it
        inpt = input_a
        if bool(random.getrandbits(1)):
            weights, firings = process_input_neurons(weights, inpt, time, train=False)
        else:
            inpt = input_b
            weights, firings = process_input_neurons(weights, inpt, time, train=False)
        
        # Tally the firings for each category
        a_neurons = 0
        b_neurons = 0
        for neuron in range(len(assignments)):
            if assignments[neuron] == 1:
                a_neurons += firings[neuron]
            elif assignments[neuron] == 0:
                b_neurons += firings[neuron]
            else: 
                print("Expected 1 or 0, got: " + str(assignments[neuron]))

        # Adjust firings for number of neurons in that category
        num_a = np.count_nonzero(assignments == 1)
        num_b = np.count_nonzero(assignments == 0)
        if num_a and num_b:
            a_neurons /= float(num_a)
            b_neurons /= float(num_b)
        else:
            a_neurons = 0
            b_neurons = 0

        # Did the right ones fire?
        if a_neurons >= b_neurons and np.array_equal(inpt, input_a):
            right += 1
        elif a_neurons < b_neurons and np.array_equal(inpt, input_b):
            right += 1
        else:
            wrong += 1

        # Calculate accuracy
        accuracy = 0
        if right and not wrong:
            accuracy = 1.0
        else:
            accuracy = right / (right + wrong)
    
    return accuracy


def run(train_iters, time):
    # Genewrate random initial weights between zero and 1
    weights = initiate_weights()
    
    # Train the weights and calculate the neuron assignments 
    weights, assignments = train(train_iters, time, weights)

    # Test model
    print("Final Accuracy: " + str(test(time, weights, assignments)))

        
run(train_iters, time)
