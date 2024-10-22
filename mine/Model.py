import numpy as np
from mine.module import Module

class Model():
    def forward(self, x:np.ndarray) -> np.ndarray:
        raise NotImplementedError()
    
    def backward(self, grad:np.ndarray, learning_rate:float) -> np.ndarray:
        raise NotImplementedError()
    
class Sequantial(Model):
    def __init__(self, modules:list[Module]) -> None:
        self.modules = modules

    def forward(self, x:np.ndarray) -> np.ndarray:
        for module in self.modules:
            x = module.forward(x)
        return x
    
    def backward(self, grad:np.ndarray, learning_rate:float):
        for module in reversed(self.modules):
            grad = module.backward(grad, learning_rate)