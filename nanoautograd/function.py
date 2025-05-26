from typing import Any, Dict, Tuple
import numpy as np

class Function:
    "Base class for all autograd operations"
    @staticmethod
    def forward(ctx:Dict[str,Any], *args:Any, **kwargs:Any) -> Any:
        raise NotImplementedError("Subclasses must implement forward.")
    
    @staticmethod
    def backward(ctx:Dict[str, Any], grad_out: np.ndarray) -> Tuple:
        raise NotImplementedError("Subclasses must implement backward.")
    
    @classmethod
    def apply(cls, *args, **kwargs):
        from .tensor import Tensor
        requires_grad = any(
            isinstance(arg, Tensor) and arg.requires_grad
            for arg in args
        )
        ctx = {}
        tensor_args = [arg.data if isinstance(arg, Tensor) else arg for arg in args]
        
        # forward pass
        output_data = cls.forward(ctx, *tensor_args, **kwargs)

        # create output tensor
        output = Tensor(
            data = output_data,
            _op = cls(),
            _children = set(arg for arg in args if isinstance(arg, Tensor)),
            requires_grad = requires_grad
        )
        
        def _backward():
            grads = cls.backward(ctx, output.grad)
            if not isinstance(grads, tuple):
                grads = (grads,)
            for arg, grad in zip(args, grads):
                if isinstance(arg, Tensor) and arg.requires_grad:
                    if grad is not None:
                        if arg.grad is None:
                            arg.grad = np.zeros_like(arg.data)
                        arg.grad = arg.grad + grad

        output._backward = _backward  
        return output                    