class Optimizer:
    def __init__(self, learning_rate: float):
        self.learning_rate = learning_rate

    def step(self, params, grads):
        raise NotImplementedError()
    
    def copy(self):
        raise NotImplementedError()

class AdamOptimizer(Optimizer):
    def __init__(self, learning_rate):
        super().__init__(learning_rate)

    def step(self, params, grads):
        return super().step(params, grads)

class SGDOptimizer(Optimizer):
    def __init__(self, learning_rate):
        super().__init__(learning_rate)

    def step(self, params, grads):
        for param, grad in zip(params, grads):
            param -= self.learning_rate * grad

    def copy(self):
        return SGDOptimizer(self.learning_rate)
