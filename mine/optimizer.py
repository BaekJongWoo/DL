import numpy as np

class Optimizer:
    def __init__(self, learning_rate: float):
        self.learning_rate = learning_rate

    def step(self, params, grads):
        raise NotImplementedError()
    
    def copy(self):
        raise NotImplementedError()

class AdamOptimizer(Optimizer):
    def __init__(self, learning_rate, beta1=0.9, beta2=0.999, epsilon=1e-8):
        super().__init__(learning_rate)
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = {}
        self.v = {}
        self.t = 0

    def step(self, params, grads):
        if not self.m:
            for i, param in enumerate(params):
                self.m[i] = np.zeros_like(param)
                self.v[i] = np.zeros_like(param)

        max_norm = 1.0
        grads = [np.clip(grad, -max_norm, max_norm) for grad in grads]

        self.t += 1

        for i, (param, grad) in enumerate(zip(params, grads)):
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * grad
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (grad ** 2)

            m_hat = self.m[i] / (1 - self.beta1 ** self.t)
            v_hat = self.v[i] / (1 - self.beta2 ** self.t)

            param -= self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)

    def copy(self):
        return AdamOptimizer(self.learning_rate, self.beta1, self.beta2, self.epsilon)

class SGDOptimizer(Optimizer):
    def __init__(self, learning_rate):
        super().__init__(learning_rate)

    def step(self, params, grads):
        for param, grad in zip(params, grads):
            param -= self.learning_rate * grad

    def copy(self):
        return SGDOptimizer(self.learning_rate)
