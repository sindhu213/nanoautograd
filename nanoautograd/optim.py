from .tensor import Tensor
import numpy as np

class Optimizer:
    def __init__(self, parameters: list[Tensor]):
        self.parameters = parameters

    def zero_grad(self,):
        for param in self.parameters:
            param.zero_grad()

    def step(self):
        raise NotImplementedError("Subclasses must implement step.") 

class SGD(Optimizer):
    def __init__(
            self, 
            parameters: list[Tensor], 
            learning_rate:float = 0.01, 
            momemtum: float = 0.0,
        ):
        
        super().__init__(parameters)
        self.learning_rate = learning_rate
        self.momemtum = momemtum
        self.velocity = {}

        for i, param in enumerate(self.parameters):
            self.velocity[i] = np.zeros_like(param.data)

    def step(self):
        for i, param in enumerate(self.parameters):
            if param.grad is None:
                continue
            self.velocity[i] = self.momemtum * self.velocity[i] - self.learning_rate * param.grad
            param.data = param.data + self.velocity[i]

class Adam(Optimizer):
    def __init__(
            self, 
            parameters: list[Tensor], 
            learning_rate:float = 0.001, 
            betas: tuple = (0.9, 0.999),
            eps: float = 1e-8
        ):

        super().__init__(parameters)
        self.betas = betas
        self.learning_rate = learning_rate
        self.eps = eps
        self.t = 0

        self.m = {}  # first moment
        self.v = {}  # second moment

        for i, param in enumerate(self.parameters):
            self.m[i] = np.zeros_like(param.data)
            self.v[i] = np.zeros_like(param.data)

    def step(self):
        self.t += 1
        beta1, beta2 = self.betas
        for i, param in enumerate(self.parameters):
            if param.grad is None:
                continue
            self.m[i] = beta1 * self.m[i] + (1-beta1) * param.grad
            self.v[i] = beta2 * self.v[i] + (1-beta2) * param.grad * param.grad
            m_hat = self.m[i] / (1-beta1 ** self.t) 
            v_hat = self.v[i] / (1-beta2 ** self.t) 
            param.data = param.data - self.learning_rate * m_hat /( np.sqrt(v_hat) + self.eps)