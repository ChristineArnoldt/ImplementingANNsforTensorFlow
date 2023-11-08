"""A manual implementation of a multi-layer perceptron for digit-classification using only numpy.

Implementation of a multi-layer-perceptron to classify the digits dataset from sklearn including an
implementation of minibatches, a sigmoid activation function for the hidden layers and a softmax
activation function for the output layer and a categorical cross entropy function.
"""

import numpy as np
import sklearn.datasets

MINIBATCHSIZE = 1

class SigmoidActivationFunction:
    """The SigmoidActivationFunction class is a manual implementation of the sigmoid function using
    numpy.
    
    It checks the shape of the preactivation (input) to be valid and sets the activations after the
    function is applied to the preactivation.

    Attributes:
        preactivation: Numpy array of size (minibatchsize, num_units) with preactivation matrix.
        num_units: Integer that specifies 
    """
    def __init__(self, preactivation: np.array, num_units: int):
        if preactivation.shape != (MINIBATCHSIZE, num_units):
            # check size of preactivation matrix and raise error if incorrect
            raise ValueError(f"Numpy Array of shape ({MINIBATCHSIZE}, {num_units}) expected.\
                Got shape {preactivation.shape}")
        self.preactivation = preactivation
        self.activations = np.empty(shape=preactivation.shape)

    def call(self):
        """Call function to calculate activations.

        Calculates a matrix with activations using the sigmoid function.
        """
        # get index of row and column of each element & calculate the activation.
        for row_idx, _ in enumerate(self.preactivation):
            for col_idx, _ in enumerate(self.preactivation[0]):
                self.activations[row_idx][col_idx] = 1 / (1 + np.exp(-self.preactivation[row_idx][col_idx]))

    def backpropagation(self):
        print('test')

class SoftmaxActivationFunction:
    """The SoftmaxActivationFunction class is a manual implementation of the softmax function using
    numpy.
    
    It checks the shape of the preactivation (input) to be valid and sets the activations after the
    function is applied to the preactivation. It's used for the final layer of the network.

    Attributes:
        preactivation: Numpy array of size (minibatchsize, 10) with preactivation matrix where 10 is
        the number of classes.
    """
    def __init__(self, preactivation: np.array):
        if preactivation.shape != (MINIBATCHSIZE, 10):
            # check size of preactivation matrix and raise error if incorrect
            raise ValueError(f"Numpy Array of shape ({MINIBATCHSIZE}, 10) expected. \
                Got shape {preactivation.shape}")
        self.preactivation = preactivation
        self.activations = np.empty(shape=preactivation.shape)

    def call(self):
        """Call function to calculate activations.

        Calculates a matrix with activations using the softmax function.
        """
        for row_idx, _ in enumerate(self.preactivation):
            for col_idx, _ in enumerate(self.preactivation[0]):
                self.activations[row_idx][col_idx] = np.exp(self.preactivation[row_idx][col_idx]) / np.sum(np.exp(self.preactivation[row_idx]))
    
    def backpropagation(self):
        print('test')

class MLPLayer:
    """A single layer for a multi layer perceptron (MLP) with forward and backward pass.

    Creates a single layer for a multi layer perceptron, checks the input size, calculates
    preactivation from weights and biases, applies the activation function.


    Attributes:
        activation_func: A string (either 'softmax' or 'sigmoid') that specifies the activation function of the layer.
        num_units: An integer that specifies the number of perceptrons in the layer.
        input_size: An integer that specifies the number of perceptrons in the previous layer.
    """
    def __init__(self, activation_func: str, num_units: int, input_size: int):
        self.activation_func = activation_func
        self.num_units = num_units
        self.input_size = input_size

    def forward_pass(self, activations: np.array) -> np.array:
        """Forward pass in the MLP layer.

        Checks input matrix from previous layer, calculates preactivation from weights and biases
        and applies activation function.

        Args:
        activations:
            Numpy array with activations (output) from previous layer.
        
        Returns:
        output:
            An output matrix with the new activations that are returned and can be used as inputs
            for subsequent layers.

        Raises:
        ValueError: An error when numpy arrays are the incorrect size.
        """

        if activations.shape != (MINIBATCHSIZE, self.input_size):
            raise ValueError(f"Numpy Array of shape ({MINIBATCHSIZE}, {self.input_size}) expected. Got shape {activations.shape}")
        
        # adding 1s at the end of every row for calculations so bias can be incorporated in weight matrix
        activations = np.column_stack([activations, 1])

        weights = np.random.normal(loc=0.0, scale=0.2, size=(len(activations[0])-1, self.num_units))
        
        bias = np.zeros(shape=(1,self.num_units))
        
        weights = np.vstack([weights,bias])
        
        if activations.shape != (MINIBATCHSIZE, self.input_size+1):
            raise ValueError(f"Numpy Array of shape ({MINIBATCHSIZE}, {self.input_size+1}) expected. Got shape {activations.shape}")
        if weights.shape != (self.input_size+1, self.num_units):
            raise ValueError(f"Numpy Array of shape ({self.input_size+1}, {self.num_units}) expected. Got shape {weights.shape}")
        
        preactivation = np.matmul(activations,weights)
        
        if preactivation.shape != (MINIBATCHSIZE, self.num_units):
            raise ValueError(f"Numpy Array of shape ({MINIBATCHSIZE}, {self.num_units}) expected. Got shape {preactivation.shape}")
        
        if self.activation_func == "softmax":
            softmax = SoftmaxActivationFunction(preactivation = preactivation)
            softmax.call()
            output = softmax.activations
        elif self.activation_func == "sigmoid":
            sigmoid = SigmoidActivationFunction(preactivation = preactivation, num_units = self.num_units)
            sigmoid.call()
            output = sigmoid.activations
        
        if output.shape != (MINIBATCHSIZE, self.num_units):
            raise ValueError(f"Numpy Array of shape ({MINIBATCHSIZE}, {self.num_units}) expected. Got shape {output.shape}")
        else:
            return output

class MLP:
    def __init__(self, num_layers: int, units: list, input: np.array, target: np.array):
        self.num_layers = num_layers
        self.units = units
        self.input = input
        self.target = target
        self.layers = []
    
    def call(self):
        for i in range(0,self.num_layers-1):
            layer = MLPLayer(activation_func="sigmoid", num_units=self.units[i], input_size=len(self.input[0]))
            self.layers.append(layer)
            self.input = layer.forward_pass(self.input)
        
        output_layer = MLPLayer(activation_func="softmax", num_units=self.units[self.num_layers-1], input_size=len(self.input[0]))   
        self.layers.append(output_layer)
        return layer.forward_pass(self.input)
    
class CategoricalCrossCntropy:
    def __init__(self, input: np.array, target: np.array):
        self.input = input
        self.target = target
        self.loss = np.nan
    
    def call(self):
        losses = []
        for input,target in zip(input,target):
            # adding a tiny number to the input (10**-10000) ensures numerical stability in case the input is 0 (numpy returns -inf for log(0))
            loss = -np.sum(target * np.log(input + 10**-10000))
            losses.append(loss)
        self.loss = np.sum(losses)
        return self.loss
    
    def backpropagation(self):
        if self.input.shape != (MINIBATCHSIZE, 10):
            raise ValueError(f"Numpy Array of shape ({MINIBATCHSIZE}, 10) expected. Got shape {self.input.shape}")
        if self.loss.shape != (MINIBATCHSIZE, 1):
            raise ValueError(f"Numpy Array of shape ({MINIBATCHSIZE}, 1) expected. Got shape {self.loss.shape}")
        
        
        
        
        #if output.shape != (MINIBATCHSIZE, 1):
        #    raise ValueError(f"Numpy Array of shape ({MINIBATCHSIZE}, 1) expected. Got shape {output.shape}")

def one_hot_encoding(targets,num_classes):
    one_hot_encodings = np.eye(num_classes)[targets]
    return one_hot_encodings

def generate_minibatches(minibatchsize): 
    '''
    Generates 2 ndarrays for input and target with minibatchsize many entries
    '''
    if minibatchsize > 1797 or minibatchsize <= 0:
        raise ValueError("Invalid minibatchsize. Expected batch size between 1 and 1797")

    # extract data
    inputs, targets = sklearn.datasets.load_digits(return_X_y=True)
    
    # convert input data to float 32 and use a min-max scaler to scale values to be in range [0,1]
    inputs = np.float32(inputs)
    inputs = (inputs-np.min(inputs))/(np.max(inputs)-np.min(inputs))
    targets = one_hot_encoding(targets, num_classes=10)
    
    # zip inputs and targets to list
    pairs = list(zip(inputs,targets))
    # shuffle tuples
    np.random.shuffle(pairs)

    # minibatches
    input_minibatch = []
    target_minibatch = []
    for i in range(minibatchsize):
        input_minibatch.append(pairs[i][0])
        target_minibatch.append(pairs[i][1])
    
    input_minibatch = np.array(input_minibatch)
    target_minibatch = np.array(target_minibatch)

    return input_minibatch,target_minibatch


def main():
    np.random.seed(42)
    input,target  = generate_minibatches(MINIBATCHSIZE)
    mlp = MLP(num_layers=5, units=[5,4,4,4,3], input = input, target = target)
    print(mlp.call())

if __name__ == "__main__":
    main()
