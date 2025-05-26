import numpy as np
from typing import Union
from .function import Function
from .tensor import Tensor

def _ensure_tensor(x:Union[int, float, np.ndarray, Tensor]) -> Tensor:
    "Convert input to a tensor, if not already."
    if isinstance(x, Tensor):
        return x
    return Tensor(x)

# grad_out is equivalent to upstream gradient
# ctx: context dictionary (stores info needed for backward pass)
class Add(Function):
    @staticmethod
    def forward(ctx, x, y):
        return x+y
    
    @staticmethod
    def backward(ctx,grad_out):
        return grad_out, grad_out

class Mul(Function):
    @staticmethod
    def forward(ctx, x, y):
        ctx['x'] = x
        ctx['y'] = y
        return x * y
    
    @staticmethod
    def backward(ctx, grad_out):
        return ctx['x'] * grad_out, ctx['y'] * grad_out

class Neg(Function):
    @staticmethod
    def forward(ctx,x):
        return -1 * x
    
    @staticmethod
    def backward(ctx, grad_out):
        return -1 * grad_out

class Sub(Function):
    @staticmethod
    def forward(ctx, x, y):
        return x - y
    
    @staticmethod
    def backward(ctx, grad_out):
        return grad_out, -1 * grad_out

class Div(Function):
    @staticmethod
    def forward(ctx, x, y):
        ctx['x'] = x
        ctx['y'] = y
        return x/y
    
    @staticmethod
    def backward(ctx, grad_out):
        x, y = ctx['x'], ctx['y']
        return grad_out/y, -1 * (x*grad_out)/(y*y)

class Pow(Function):
    @staticmethod
    def forward(ctx, x, y):
        ctx['x'] = x
        ctx['y'] = y
        return np.power(x,y)
    
    @staticmethod
    def backward(ctx, grad_out):
        x, y = ctx['x'], ctx['y']
        grad_x = y*np.power(x,y-1)
        grad_y = None
        eps = 1e-8
        if isinstance(y, np.ndarray):
            safe_value = x.clip(eps, y-eps)
            grad_y = np.power(x, y) * np.log(safe_value) * grad_out
        return grad_x, grad_y

class MatMul(Function):
    @staticmethod
    def forward(ctx, x, y):
        ctx['x'] = x
        ctx['y'] = y
        return x@y
    
    @staticmethod
    def backward(ctx, grad_out):
        x, y = ctx['x'], ctx['y']
        grad_x = grad_out@y.T
        grad_y = x.T@grad_out
        return grad_x, grad_y

class Sum(Function):
    @staticmethod
    def forward(ctx, x, dim = None):
        ctx['dim'] = dim
        ctx['input_shape'] = x.shape
        return np.sum(x, axis=dim)
    
    @staticmethod
    def backward(ctx, grad_out):
        dim = ctx['dim']
        input_shape = ctx['input_shape']
        if dim is None:
            return np.ones(input_shape) * grad_out
        grad_input = np.expand_dims(grad_out, axis = dim)
        return np.broadcast_to(grad_input, input_shape)

class Mean(Function):
    @staticmethod
    def forward(ctx, x, dim = None):
        ctx['dim'] = dim
        ctx['input_shape'] = x.shape
        if dim is None:
            ctx['size'] = np.prod(x.shape) 
            return np.mean(x)
        else:
            ctx['size'] = x.shape[dim]
            return np.mean(x, axis = dim)
     
    @staticmethod
    def backward(ctx, grad_out):
        dim = ctx['dim']
        input_shape = ctx['input_shape']
        size = ctx['size']
        if dim is None:
            return np.ones(input_shape)*grad_out/size
        grad_input = np.expand_dims(grad_out, dim)
        return np.broadcast_to(grad_input, input_shape)/size

class Exp(Function):
    @staticmethod
    def forward(ctx, x):
        out = np.exp(x)
        ctx['out'] = out
        return out
    
    @staticmethod
    def backward(ctx, grad_out):
        out = ctx['out']
        return out * grad_out

class Log(Function):
    @staticmethod
    def forward(ctx, x):
        ctx['x'] = x
        return np.log(x)
    
    @staticmethod
    def backward(ctx, grad_out):
        x = ctx['x']
        return grad_out / x

class ReLU(Function):
    @staticmethod
    def forward(ctx, x):
        ctx['mask'] = x > 0
        return np.maximum(x, 0)
    
    @staticmethod
    def backward(ctx, grad_out):
        return grad_out * ctx['mask']

class Sigmoid(Function):
    @staticmethod
    def forward(ctx, x):
        out = 1.0 / (1.0 + np.exp(-x))
        ctx['out'] = out
        return out

    @staticmethod
    def backward(ctx, grad_out):
        out = ctx['out']
        return out * (1- out) * grad_out

class Tanh(Function):
    @staticmethod
    def forward(ctx, x):
        out = np.tanh(x)
        ctx['out'] = out
        return out

    @staticmethod
    def backward(ctx, grad_out):
        out = ctx['out']
        return (1 - out * out) * grad_out

# Wrapper functions for easier use
def add(x, y):
    return Add.apply(_ensure_tensor(x), _ensure_tensor(y))

def mul(x, y):
    return Mul.apply(_ensure_tensor(x), _ensure_tensor(y))

def neg(x):
    return Neg.apply(_ensure_tensor(x))

def sub(x, y):
    return Sub.apply(_ensure_tensor(x), _ensure_tensor(y))

def div(x, y):
    return Div.apply(_ensure_tensor(x), _ensure_tensor(y))

def pow(x, y):
    return Pow.apply(_ensure_tensor(x), _ensure_tensor(y))

def matmul(x, y):
    return MatMul.apply(_ensure_tensor(x), _ensure_tensor(y))

def sum(x, dim=None):
    return Sum.apply(_ensure_tensor(x), dim)

def mean(x, dim=None):
    return Mean.apply(_ensure_tensor(x), dim)

def exp(x):
    return Exp.apply(_ensure_tensor(x))

def log(x):
    return Log.apply(_ensure_tensor(x))

def relu(x):
    return ReLU.apply(_ensure_tensor(x))

def sigmoid(x):
    return Sigmoid.apply(_ensure_tensor(x))

def tanh(x):
    return Tanh.apply(_ensure_tensor(x))