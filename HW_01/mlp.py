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
                self.activations[row_idx][col_idx] = 1 / (1 + \
                    np.exp(-self.preactivation[row_idx][col_idx]))

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
                self.activations[row_idx][col_idx] = np.exp(self.preactivation[row_idx][col_idx]) /\
                    np.sum(np.exp(self.preactivation[row_idx]))

    def backpropagation(self):
        print('test')

class MLPLayer:
    """A single layer for a multi layer perceptron (MLP) with forward and backward pass.

    Creates a single layer for a multi layer perceptron, checks the input size, calculates
    preactivation from weights and biases, applies the activation function.


    Attributes:
        activation_func: 'softmax' or 'sigmoid' (str) specifying the activation function.
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
            # check size of preactivation matrix and raise error if incorrect
            raise ValueError(f"Numpy Array of shape ({MINIBATCHSIZE}, {self.input_size}) expected.\
                Got shape {activations.shape}")

        # adding 1s at the end of every row to enable calculations where bias is
        # incorporated in the weight matrix
        activations = np.column_stack([activations, 1])
        # setting random weights with a mu=0 and sigma=0.2
        weights = np.random.normal(loc=0.0, scale=0.2, size=(len(activations[0])-1, self.num_units))
        # creating bias vector with zeros
        bias = np.zeros(shape=(1,self.num_units))
        # incorporating biases in the weight matrix by stacking the arrays vertically
        weights = np.vstack([weights,bias])

        if activations.shape != (MINIBATCHSIZE, self.input_size+1):
            # check size of activation matrix (inputs) and raise error if incorrect
            raise ValueError(f"Numpy Array of shape ({MINIBATCHSIZE}, {self.input_size+1})\
                expected. Got shape {activations.shape}")
        if weights.shape != (self.input_size+1, self.num_units):
            # check size of weight matrix and raise error if incorrect
            raise ValueError(f"Numpy Array of shape ({self.input_size+1}, {self.num_units})\
                expected. Got shape {weights.shape}")

        # calculate preactivations by doing matrix multiplication of the activations from the
        # previous layer (input) matrix and the weights matrix (with bias incorporated)
        preactivation = np.matmul(activations,weights)

        if preactivation.shape != (MINIBATCHSIZE, self.num_units):
            # check size of preactivation matrix and raise error if incorrect
            raise ValueError(f"Numpy Array of shape ({MINIBATCHSIZE}, {self.num_units}) expected.\
                Got shape {preactivation.shape}")

        # check what the specified activation function for the layer is and apply the function
        if self.activation_func == "softmax":
            softmax = SoftmaxActivationFunction(preactivation = preactivation)
            softmax.call()
            output = softmax.activations
        elif self.activation_func == "sigmoid":
            sigmoid = SigmoidActivationFunction(preactivation = preactivation,\
                num_units = self.num_units)
            sigmoid.call()
            output = sigmoid.activations

        if output.shape != (MINIBATCHSIZE, self.num_units):
            # check size of output matrix (new activations) and raise error if incorrect
            raise ValueError(f"Numpy Array of shape ({MINIBATCHSIZE}, {self.num_units})\
                expected. Got shape {output.shape}")

        return output

class MLP:
    """Multi layer perceptron.

    A multi layer perceptron with a specified number of layers with a specified number of
    perceptrons.

    Attributes:
        num_layers: An integer specifying the number of layers.
        units: A list of integers
        input_data: Numpy array matrix with minibatched input data.
        targets: Numpy array with one hot encoded targets
        layers: List of layers created.
    """
    def __init__(self, num_layers: int, units: list[int], input_data: np.array, targets: np.array):
        self.num_layers = num_layers
        self.units = units
        self.input_data = input_data
        self.targets = targets
        self.layers = []

    def call(self) -> np.array:
        """Creates a multi layer perceptron with a specified number of layers with a specified
        number of perceptrons.

        Creates layers and calls the forward- and backward-pass functions.

        Returns:
        Returns the output of the forward pass function of the layer with the calculated activations
        matrix as a numpy array.
        """
        # for all layers (except the final one) create a layer object with a sigmoid activation,
        # append the layer to the list of layers and do a forward pass
        for i in range(0,self.num_layers-1):
            layer = MLPLayer(activation_func="sigmoid", num_units=self.units[i],\
                input_size=len(self.input_data[0]))
            self.layers.append(layer)
            self.input_data = layer.forward_pass(self.input_data)

        # create the output layer with a softmax activation function, append the layer to the
        # list of layers and do a forward pass
        output_layer = MLPLayer(activation_func="softmax", num_units=self.units[self.num_layers-1],\
            input_size=len(self.input_data[0]))   
        self.layers.append(output_layer)
        return layer.forward_pass(self.input_data)

class CategoricalCrossEntropy:
    """Calculate the categorical cross entropy.

    Attributes:
        input_data: A numpy array with the inputs.
        targets: A numpy array with the target classes.
    """
    def __init__(self, input_data: np.array, targets: np.array):
        self.input_data = input_data
        self.targets = targets
        self.loss = np.nan

    def call(self):
        """Calculate the categorical cross entropy loss

        Returns:
        The calculated loss for the minibatch.
        """
        losses = []
        # match inputs and targets, calculate individual loss and append to list, sum all losses
        for input_data,targets in zip(input_data,targets):
            # adding a tiny number to the input_data (10**-10000) ensures numerical stability in
            # case the input_data is 0 (numpy returns -inf for log(0))
            loss = -np.sum(targets * np.log(input_data + 10**-10000))
            losses.append(loss)
        self.loss = np.sum(losses)
        return self.loss

    def backpropagation(self):
        """Runs the backpropagation and adjusts the weights.

        Raises:
        ValueError: An error when the numpy array shape is not the expected size.
        """
        if self.input_data.shape != (MINIBATCHSIZE, 10):
            raise ValueError(f"Numpy Array of shape ({MINIBATCHSIZE}, 10) expected.\
                Got shape {self.input_data.shape}")
        if self.loss.shape != (MINIBATCHSIZE, 1):
            raise ValueError(f"Numpy Array of shape ({MINIBATCHSIZE}, 1) expected.\
                Got shape {self.loss.shape}")

        #if output.shape != (MINIBATCHSIZE, 1):
        #    raise ValueError(f"Numpy Array of shape ({MINIBATCHSIZE}, 1) expected.\
            # Got shape {output.shape}")

def one_hot_encoding(targets: np.array,num_classes) -> np.array:
    """Creates one hot encodings of target numpy array.

    Args:
      targets:
        Numpy array with target classes.
      num_classes:
        A number of classes .
      require_all_keys:
        If True only rows with values set for all keys will be returned.

    Returns:
      A dict mapping keys to the corresponding table row data
      fetched. Each row is represented as a tuple of strings. For
      example:

      {b'Serak': ('Rigel VII', 'Preparer'),
       b'Zim': ('Irk', 'Invader'),
       b'Lrrr': ('Omicron Persei 8', 'Emperor')}

      Returned keys are always bytes.  If a key from the keys argument is
      missing from the dictionary, then that row was not found in the
      table (and require_all_keys must have been False).

    Raises:
      IOError: An error occurred accessing the smalltable.
    """
    one_hot_encodings = np.eye(num_classes)[targets]
    return one_hot_encodings

def generate_minibatches(minibatchsize):
    """Generate minibatches for training.

    Generates 2 numpy arrays for input and target with minibatchsize many entries

    Args:
      minibatchsize:
        Maximum number of elements in minibatches.

    Returns:
      input_minibatch:
        numpy array with minibatches of input data
      target_minibatch:
        numpy array with minibatches of target data (labels)

    Raises:
      ValueError: An error when the minibatchsize is larger than the dataset or <=0 (invalid size)
    """
    # extract data
    inputs, targets = sklearn.datasets.load_digits(return_X_y=True)

    if minibatchsize > inputs.shape[0] or minibatchsize <= 0:
        # raise error if minibatchsize larger than dataset or negative or 0
        raise ValueError("Invalid minibatchsize. Expected batch size between 1 and 1797")

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
    inputs,targets  = generate_minibatches(MINIBATCHSIZE)
    mlp = MLP(num_layers=5, units=[5,4,4,4,3], input_data = inputs, targets = targets)
    print(mlp.call())

if __name__ == "__main__":
    main()
