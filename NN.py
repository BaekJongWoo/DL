import numpy as np
from Base import MyModule, MyModel

class linear(MyModule):
    def __init__(self, input_size:int, output_size:int, learning_rate:float=0.01) -> None:
        self.W = np.random.rand(output_size, input_size)
        self.b = np.random.rand(output_size, 1)

        self.x = None
        self.lr = learning_rate

    def forward(self, input):
        self.x = input
        return np.dot(self.W, input) + self.b   
    
    def backward(self, output_gradient):
        dx = np.dot(self.W.T, output_gradient)
        dW = np.dot(output_gradient, self.x.T)
        db = output_gradient
        
        self.W -= self.learning_rate * dW
        self.b -= self.learning_rate * db

        return dx

class ReLU(MyModule):
    def __init__(self) -> None:
        self.x

    def forward(self, input):
        self.x = input
        return np.max(0, input)
    
    def backward(self, output_gradient):
        return float(self.x > 0) * output_gradient
    
class NN(MyModel):
    def __init__(self) -> None:
        input_size = 28*28
        h1_size = 256
        h2_size = 128
        output_size = 10

        self.linear1 = linear(input_size, h1_size)
        self.linear2 = linear(h1_size, h2_size)
        self.linear3 = linear(h2_size, output_size)
        self.relu = ReLU()

    def forward(self, input):
        x = input
        x = linear.forward(x)
        