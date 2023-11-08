import random
import sys
import numpy as np
import sklearn.datasets
import math

MINIBATCHSIZE = 1

class SigmoidActivationFunction:
    def __init__(self, input: np.array, num_units: int):
        if input.shape != (MINIBATCHSIZE, num_units):
            raise ValueError(f"Numpy Array of shape ({MINIBATCHSIZE}, {num_units}) expected. Got shape {input.shape}")
        self.input = input
        self.activations = np.empty(shape=input.shape)

    def call(self):
        for row_idx in range(len(self.input)):
            for col_idx in range(len(self.input[0])):
                self.activations[row_idx][col_idx] = 1 / (1 + math.exp(-self.input[row_idx][col_idx]))
                
    def backpropagation(self):
        print('test')

class SoftmaxActivationFunction:
    def __init__(self, input: np.array):
        if input.shape != (MINIBATCHSIZE, 10):
            raise ValueError(f"Numpy Array of shape ({MINIBATCHSIZE}, 10) expected. Got shape {input.shape}")
        self.input = input
        self.activations = np.empty(shape=input.shape)

    def call(self):
        for row_idx in range(len(self.input)):
            for col_idx in range(len(self.input[0])):
                self.activations[row_idx][col_idx] = np.exp(self.input[row_idx][col_idx]) / np.sum(np.exp(self.input[row_idx]))
    
    def backpropagation(self):
        print('test')

class MLPLayer:
    def __init__(self, activation_func, num_units, input_size):
        self.activation_func = activation_func
        self.num_units = num_units
        self.input_size = input_size
    
    def forward_pass(self, input: np.array) -> np.array:
        if input.shape != (MINIBATCHSIZE, self.input_size):
            raise ValueError(f"Numpy Array of shape ({MINIBATCHSIZE}, {self.input_size}) expected. Got shape {input.shape}")
        
        # adding 1s at the end of every row for calculations so bias can be incorporated in weight matrix
        input = np.column_stack([input, 1])

        weights = np.random.normal(loc=0.0, scale=0.2, size=(len(input[0])-1, self.num_units))
        
        bias = np.zeros(shape=(1,self.num_units))
        
        weights = np.vstack([weights,bias])
        
        if input.shape != (MINIBATCHSIZE, self.input_size+1):
            raise ValueError(f"Numpy Array of shape ({MINIBATCHSIZE}, {self.input_size+1}) expected. Got shape {input.shape}")
        if weights.shape != (self.input_size+1, self.num_units):
            raise ValueError(f"Numpy Array of shape ({self.input_size+1}, {self.num_units}) expected. Got shape {weights.shape}")
        
        preactivation = np.matmul(input,weights)
        
        if preactivation.shape != (MINIBATCHSIZE, self.num_units):
            raise ValueError(f"Numpy Array of shape ({MINIBATCHSIZE}, {self.num_units}) expected. Got shape {preactivation.shape}")
        
        if self.activation_func == "softmax":
            softmax = SoftmaxActivationFunction(input = preactivation)
            softmax.call()
            output = softmax.activations
        elif self.activation_func == "sigmoid":
            sigmoid = SigmoidActivationFunction(input = preactivation, num_units = self.num_units)
            sigmoid.call()
            output = sigmoid.activations
        
        if output.shape != (MINIBATCHSIZE, self.num_units):
            raise ValueError(f"Numpy Array of shape ({MINIBATCHSIZE}, {self.num_units}) expected. Got shape {output.shape}")



def generate_minibatches(minibatchsize): 
    '''
    Generates 2 ndarrays for input and target with minibatchsize many entries
    '''
    if(minibatchsize > 1797):
        print('Like does this number sound mini to you? Give a minibatchsize that is lower or equal to 1797.')
        sys.exit(0)

    # variable in which tuples will be collected in
    tuples = []

    # extract data
    data = sklearn.datasets.load_digits(return_X_y=True)

    # form tuples and reshape input and target formats
    for i in range(len(data[0])): # 1797 samples
        input = data[0][i]
        for j in range(len(input)): # 64 entries
            input[j] = np.float32(input[j]/16) # represented as float32 with [0 to 1]
        
        target = data[1][i]
        layout = [0,0,0,0,0,0,0,0,0,0]
        layout[target] = 1
        target = layout

        # create list of tuples
        tuple = (input, target)
        tuples.append(tuple)

    # shuffle tuples
    random.shuffle(tuples)

    # minibatches
    input_minibatch = []
    target_minibatch = []
    for m in range(minibatchsize):
        input_minibatch.append(tuples[m][0])
        target_minibatch.append(tuples[m][1])
    
    input_minibatch = np.array(input_minibatch)
    target_minibatch = np.array(target_minibatch)

    return input_minibatch,target_minibatch


def main():
    input,target  = generate_minibatches(MINIBATCHSIZE)
    layer01 = MLPLayer(activation_func="sigmoid", num_units=3, input_size=len(input[0]))
    layer01.forward_pass(input)

if __name__ == "__main__":
    main()
