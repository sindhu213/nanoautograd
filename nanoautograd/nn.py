class Module:
    def __call__(self, *args, **kwds):
        pass
    def forward(self):
        pass
    def zero_grad(self):
        pass
    def parameters(self):
        pass

class Linear(Module):
    def __init__(self):
        super().__init__()
    def forward(self):
        return 
    
class Sequential(Module):
    def __init__(self):
        super().__init__()
    def forward(self):
        return

class ReLU(Module):
    def forward(self):
        return 

class Sigmoid(Module):
    def forward(self):
        return
    
class Tanh(Module):
    def forward(self):
        return

class MSELoss(Module):
    def forward(self):
        return

class BinaryCrossEntropyLoss():
    def forward(self):
        return