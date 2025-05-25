import numpy as np
from typing import Union, List, Set, Optional, Tuple
from function import Function

class Tensor:
    def __init__(
            self,
            data: Union[int, float, np.ndarray, List], 
            _op: Optional[Function] = None,
            _children: Set = set(),
            requires_grad: bool = False
    ) -> None:
        # Type cast to correct datatype
        if isinstance(data, (int,float,List)):
            self.data = np.array(data, dtype=np.float64)
        elif isinstance(data,np.ndarray):
            self.data = data.astype(np.float64)
        else:
            raise TypeError("Unsupported data type")
        
        self.requires_grad = requires_grad
        self.grad = None
        self._prev = _children
        self._op = _op  
        self._backward = lambda: None

        # Turn off gradient tracking for efficiency
        if self.requires_grad:
            self.zero_grad()

    def zero_grad(self) -> None:
        self.grad = np.zeros_like(self.data, dtype=np.float64)

    @property
    def shape(self) -> Tuple:
        return self.data.shape

    def backward(self, grad: Optional[np.ndarray] = None) -> None:
        if self.requires_grad == False:
            return
        
        # Initialize gradient for scalar output nodes
        if grad == None:
            if np.prod(self.shape) == 1:
                grad = np.ones_like(self.data)
            else:
                raise RuntimeError("Gradients can only be implicitly assigned for scalar outputs")

        self.grad = grad if self.grad is None else self.grad + grad

        # Topological sorting 
        topo = []
        visited = set()
        def build_topo(node):
            if node not in visited:
                visited.add(node)
                for child in node._prev:
                    build_topo(child)
                topo.append(node)
        build_topo(self)

        # Backward pass
        for node in reversed(topo):
            if node._op:
                node._op._backward()

    def __add__(self, other):
        from .operations import add
        return add(self, other) 
    
    def __mul__(self, other):
        from .operations import mul
        return mul(self, other)
    
    def __pow__(self, other):
        from .operations import pow
        return pow(self, other)
    
    def __truediv__(self, other):
        from .operations import div
        return div(self,other)

    def __neg__(self):
        from .operations import neg
        return neg(self)
        
    def __sub__(self, other):
        from .operations import sub
        return sub(self, other)

    def __matmul__(self, other):
        from .operations import matmul
        return matmul(self, other)
    
    def sum(self, dim=None):
        from .operations import sum
        return sum(self, dim)
    
    def mean(self, dim=None):
        from .operations import mean
        return mean(self, dim)
    
    def exp(self):
        from .operations import exp
        return exp(self)
    
    def log(self):
        from .operations import log
        return log(self)
    
    def relu(self):
        from .operations import relu
        return relu(self)
    
    def sigmoid(self):
        from .operations import sigmoid
        return sigmoid(self)
    
    def tanh(self):
        from .operations import tanh
        return tanh(self)
    
    def __repr__(self) -> str:
        return f"Tensor${self.data}, requires_grad{self.requires_grad}"