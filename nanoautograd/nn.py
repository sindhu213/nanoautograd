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
        def _gather_params(obj):
            params = []
            if isinstance(obj, Tensor):
                params.append(obj)
            elif isinstance(obj, Module):
                params.extend(obj.parameters())
            elif isinstance(obj, (list, tuple)):
                for item in obj:
                    params.extend(_gather_params(item))
            elif isinstance(obj, dict):
                for item in obj.values():
                    params.extend(_gather_params(item))
            return params

        params = []
        for attr in self.__dict__.values():
            params.extend(_gather_params(attr))
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
        self.bias = None
        if bias:
            self.bias = Tensor(
                data = np.zeros(out_features),
                requires_grad=True
            )
        
    def forward(self, x: Tensor):
        out = matmul(x, self.weights)  
        if self.bias is not None:
            out = add(out, self.bias)
        return out
    
class Sequential(Module):
    def __init__(self, *layers):
        super().__init__()
        self.layers = layers

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

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