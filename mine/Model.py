import numpy as np
from mine.optimizer import Optimizer

class ModelBase():
    def set_optimizer(self, optimizer: Optimizer):
        self.optimizer = optimizer

    def forward(self, x:np.ndarray, is_train: bool) -> np.ndarray:
        raise NotImplementedError()
    
    def backward(self, grad:np.ndarray) -> np.ndarray:
        raise NotImplementedError()

class Sequantial(ModelBase):
    def __init__(self, modules:list[ModelBase], optimizer: Optimizer) -> None:
        self.modules = modules
        for module in self.modules:
            module.is_train = True
            module.set_optimizer(optimizer.copy())

    def forward(self, x:np.ndarray, is_train: bool=True) -> np.ndarray:
        for module in self.modules:
            x = module.forward(x, is_train)
        return x
    
    def backward(self, grad:np.ndarray):
        for module in reversed(self.modules):
            grad = module.backward(grad)