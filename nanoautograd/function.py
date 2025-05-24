class Function:
    "Base class for all autograd operations"
    @staticmethod
    def forward():
        raise NotImplementedError
    
    @staticmethod
    def backward():
        raise NotImplementedError
    
    @classmethod
    def apply(cls):
        raise NotImplementedError