import numpy as np

def sigmoid(x):
    return 1/(1+np.exp(-x))

def relu(x):
    return np.maximum(x, 0)

def tanh(x):
    return np.tanh(x)

def leakyrelu(x):
    return np.where(x > 0, x, x * 0.01)

def step(x):
    return np.heaviside(x, 1)

def sine(x):
    return np.sin(x)

class CTRNN():

    def __init__(self, size, a):
        self.Size = size                        # number of neurons in the circuit
        self.States = np.zeros(size)            # state of the neurons
        self.TimeConstants = np.ones(size)      # time-constant for each neuron
        self.invTimeConstants = 1.0/self.TimeConstants
        self.Biases = np.zeros(size)            # bias for each neuron
        self.Weights = np.zeros((size,size))    # connection weight for each pair of neurons
        self.Outputs = np.zeros(size)           # neuron outputs
        self.Inputs = np.zeros(size)            # external input to each neuron
        self.a = a

    def setParameters(self, weights, biases, timeconstants):
        self.Weights =  np.array(weights)
        self.Biases =  np.array(biases)
        self.TimeConstants =  np.array(timeconstants)
        self.invTimeConstants = 1.0/self.TimeConstants        

    def randomizeParameters(self):
        self.Weights = np.random.uniform(-10,10,size=(self.Size,self.Size))
        self.Biases = np.random.uniform(-10,10,size=(self.Size))
        self.TimeConstants = np.random.uniform(0.1,5.0,size=(self.Size))
        self.invTimeConstants = 1.0/self.TimeConstants

    def initializeState(self, s):
        self.Inputs = np.zeros(self.Size) 
        self.States = s
        if(self.a == 0):
            self.Outputs = sigmoid(self.States+self.Biases)
        elif(self.a == 1):
            self.Outputs = relu(np.clip(self.States+self.Biases, -10, 10))
        elif(self.a == 2):
            self.Outputs = leakyrelu(np.clip(self.States+self.Biases, -10, 10))
        elif(self.a == 3):
            self.Outputs = tanh(self.States+self.Biases)
        elif(self.a == 4):
            self.Outputs = step(self.States+self.Biases)
        elif(self.a == 5):
            self.Outputs = sine(self.States+self.Biases)

    def step(self, dt):
        netinput = self.Inputs + np.dot(self.Weights.T, self.Outputs)
        self.States += dt * (self.invTimeConstants*(-self.States+netinput))
        if(self.a == 0):
            self.Outputs = sigmoid(self.States+self.Biases)
        elif(self.a == 1):
            self.Outputs = relu(np.clip(self.States+self.Biases, -10, 10))
        elif(self.a == 2):
            self.Outputs = leakyrelu(np.clip(self.States+self.Biases, -10, 10))
        elif(self.a == 3):
            self.Outputs = tanh(self.States+self.Biases)
        elif(self.a == 4):
            self.Outputs = step(self.States+self.Biases)
        elif(self.a == 5):
            self.Outputs = sine(self.States+self.Biases) 

    def save(self, filename):
        np.savez(filename, size=self.Size, weights=self.Weights, biases=self.Biases, timeconstants=self.TimeConstants)

    def load(self, filename):
        params = np.load(filename)
        self.Size = params['size']
        self.Weights = params['weights']
        self.Biases = params['biases']
        self.TimeConstants = params['timeconstants']
        self.invTimeConstants = 1.0/self.TimeConstants
