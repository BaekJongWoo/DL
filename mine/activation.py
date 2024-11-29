import numpy as np
from mine.model import ModelBase
# from model import ModelBase

class ReLU(ModelBase):
    def forward(self, x: np.ndarray) -> np.ndarray:
        self.x = x
        return ReLU.s_forward(x)
    
    def backward(self, grad: np.ndarray) -> np.ndarray:
        return ReLU.s_backward(self.x, grad)
    
    def s_forward(x: np.ndarray) -> np.ndarray:
        return np.maximum(0, x)

    def s_backward(x: np.ndarray, grad: np.ndarray) -> np.ndarray:
        return (x > 0) * grad
    
class tanh(ModelBase):
    def forward(self, x: np.ndarray) -> np.ndarray:
        self.y = tanh.s_forward(x)
        return self.y

    def backward(self, grad: np.ndarray) -> np.ndarray:
        return tanh.s_backward(self.y, grad)
    
    def s_forward(x: np.ndarray) -> np.ndarray:
        return np.tanh(x)

    def s_backward(y: np.ndarray, grad: np.ndarray) -> np.ndarray:
        return grad * (1 - y ** 2)
    
class sigmoid(ModelBase):
    def forward(self, x):
        self.y = sigmoid.s_forward(x)
        return self.y
    
    def backward(self, grad):
        return sigmoid.s_backward(self.y, grad)

    def s_forward(x: np.ndarray):
        return 1 / (1 + np.exp(-x))

    def s_backward(y: np.ndarray, grad: np.ndarray):
        return grad * y * (1 - y)