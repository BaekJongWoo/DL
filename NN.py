import numpy as np
from Base import Module, Model

class linear(Module):
    def __init__(self, input_size:int, output_size:int) -> None:
        self.W = np.random.uniform(-1, 1, (input_size, output_size))
        self.b = np.random.uniform(-1, 1, (1, output_size))

        self.x = None

    def forward(self, x):
        self.x = x
        return np.dot(x, self.W) + self.b
    
    def backward(self, dy: np.ndarray, learning_rate: float) -> np.ndarray:
        batch_size = dy.shape[0]

        dx = np.dot(dy, self.W.T)  # (batch_size, input_size)
        dW = np.dot(self.x.T, dy) / batch_size  # (input_size, output_size)
        db = np.sum(dy, axis=0, keepdims=True) / batch_size  # (1, output_size)

        self.W -= learning_rate * dW
        self.b -= learning_rate * db

        return dx

class ReLU(Module):
    def forward(self, x: np.ndarray) -> np.ndarray:
        return np.max(0, x)
    
    def backward(self, dy: np.ndarray, learning_rate: float) -> np.ndarray:
        return float(self.x > 0) * dy
