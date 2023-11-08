import random
import sys
import numpy as np
import sklearn.datasets

MINIBATCHSIZE = 2

class SigmoidActivationFunction:
    def __init__(self, input, target):
        self.input = input
        self.target = target

    def call(self):
        print('test')

    def backpropagation(self):
        print('test')

class SoftmaxActivationFunction:
    def __init__(self, input, target):
        self.input = input
        self.target = target

    def call(self):
        print('test')

    def backpropagation(self):
        print('test')

class MLPLayer:
    def __init__(self, acti_func, no_units, input_size):
        self.acti_func = acti_func
        self.no_units = no_units
        self.input_size = input_size



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

    return input_minibatch,target_minibatch


def main():
    generate_minibatches(MINIBATCHSIZE)

if __name__ == "__main__":
    main()
