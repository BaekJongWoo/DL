import numpy as np

class Module():
    def SetConfig(self, learning_rate):
        self.learning_rate = learning_rate

    def forward(self, x:np.ndarray) -> np.ndarray:
        raise NotImplementedError()
    
    def backward(self, dy:np.ndarray, learning_rate:float) -> np.ndarray:
        raise NotImplementedError()

class Model():
    def forward(self, x:np.ndarray) -> np.ndarray:
        raise NotImplementedError()
    
    def forward(self, dy:np.ndarray, learning_rate:float) -> np.ndarray:
        raise NotImplementedError()

class Optimizer():
    pass

class Loss():
    pass