from .tensor import Tensor
import numpy as np
from .operations import matmul, add, relu, sigmoid, tanh

class Module:
    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)
    
    def forward(self, *args, **kwargs):
        raise NotImplementedError("Subclasses must implement forward.")

    def zero_grad(self):
        for param in self.parameters():
            param.zero_grad()

    def parameters(self):
        params = []
        for _ , value in self.__dict__.items():
            if isinstance(value, Tensor):
                params.append(value)
            elif isinstance(value, Module):
                params.extend(value.parameters())
            elif isinstance(value, list):
                for param in value:
                    if isinstance(param, Module):
                        params.extend(param.parameters())
        return params

class Linear(Module):
    def __init__(
        self, 
        in_features: int,
        out_features: int, 
        bias: bool = True
    ):
        
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Xavier uniform init
        limit = np.sqrt(6.0/(in_features + out_features))
        self.weights = Tensor(
            data = np.random.uniform(-limit, limit, (in_features, out_features)),
            requires_grad=True
        )
        if bias:
            self.bias = Tensor(
                data = np.zeros(out_features),
                requires_grad=True
            )
        
    def forward(self, x: Tensor):
        out = matmul(self.weights, x)
        if self.bias is not None:
            out = add(out, self.bias)
        return out
    
class Sequential(Module):
    def __init__(self, *layers):
        super().__init__()
        self.layers = layers

    def forward(self):
        for layer in self.layers:
            out = layer(out)
        return out

class ReLU(Module):
    def forward(self, x:Tensor):
        return relu(x)

class Sigmoid(Module):
    def forward(self, x:Tensor):
        return sigmoid(x)
    
class Tanh(Module):
    def forward(self, x: Tensor):
        return tanh(x)

class MSELoss(Module):
    def forward(self, pred: Tensor, target: Tensor):
        return ((pred - target)**2).mean()

class BinaryCrossEntropyLoss():
    def forward(self, pred: Tensor, target: Tensor):
        epsilon = 1e-12
        pred_ = pred.data.clip(epsilon, 1.0 - epsilon)
        loss = -(target.data * np.log(pred_) + (1 - target.data) * np.log(1 - pred_))
        return Tensor(float(np.mean(loss)))