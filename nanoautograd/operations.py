import numpy as np
from typing import Optional, Union, Tuple
from .function import Function

def _ensure_tensor():
    "Convert input to a tensor, if not already."
    pass

class Add(Function):
    @staticmethod
    def forward():
        pass
    
    @staticmethod
    def backward():
        pass

class Mul(Function):
    @staticmethod
    def forward():
        pass
    
    @staticmethod
    def backward():
        pass

class Neg(Function):
    @staticmethod
    def forward():
        pass
    
    @staticmethod
    def backward():
        pass

class Sub(Function):
    @staticmethod
    def forward():
        pass
    
    @staticmethod
    def backward():
        pass

class Div(Function):
    @staticmethod
    def forward():
        pass
    
    @staticmethod
    def backward():
        pass

class Pow(Function):
    @staticmethod
    def forward():
        pass
    
    @staticmethod
    def backward():
        pass

class MatMul(Function):
    @staticmethod
    def forward():
        pass
    
    @staticmethod
    def backward():
        pass

class Sum(Function):
    @staticmethod
    def forward():
        pass
    
    @staticmethod
    def backward():
        pass

class Mean(Function):
    @staticmethod
    def forward():
        pass
    
    @staticmethod
    def backward():
        pass

class Exp(Function):
    @staticmethod
    def forward():
        pass
    
    @staticmethod
    def backward():
        pass

class Log(Function):
    @staticmethod
    def forward():
        pass
    
    @staticmethod
    def backward():
        pass

class ReLU(Function):
    @staticmethod
    def forward():
        pass
    
    @staticmethod
    def backward():
        pass

class Sigmoid(Function):
    @staticmethod
    def forward():
       pass
    
    @staticmethod
    def backward():
        pass

class Tanh(Function):
    @staticmethod
    def forward():
        pass

    @staticmethod
    def backward():
        pass

# Wrapper functions for easier use
def add(x, y):
    pass

def mul(x, y):
    pass

def neg(x):
    pass

def sub(x, y):
    pass

def div(x, y):
    pass

def pow(x, y):
    pass

def matmul(x, y):
    pass

def sum(x, dim=None):
    pass

def mean(x, dim=None):
    pass

def exp(x):
    pass

def log(x):
    pass

def relu(x):
    pass

def sigmoid(x):
    pass

def tanh(x):
    pass