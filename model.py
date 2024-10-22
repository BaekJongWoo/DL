from Base import Model, Module
import numpy as np
import NN

class Sequantial(Model):
    def __init__(self, modules:list[Module]) -> None:
        self.modules = modules

    def forward(self, x:np.ndarray) -> np.ndarray:
        for module in self.modules:
            x = module.forward(x)
        return x

def MyNN():
    model = Sequantial([
        NN.linear(28*28, 10)
    ])
    return model