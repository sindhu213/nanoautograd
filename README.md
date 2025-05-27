# nanoautograd

nanoautograd is a lightweight Python library that provides a minimalistic automatic differentiation engine, inspired by PyTorch's autograd system. It allows for the construction of computational graphs and automatic computation of gradients via backpropagation. It supports basic tensor operations and activation functions.


## Installation

1. Clone the repository:

    ```bash
    git clone https://github.com/sindhu213/nanoautograd.git
    cd nanoautograd
    ```

2. (Optional) Create and activate a virtual environment:

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3. Install the required dependencies:

    ```bash
    pip install -r requirements.txt
    ```


## Usage

```python
from nanoautograd.tensor import Tensor

# Create input tensors
a = Tensor(2.0, requires_grad=True)
b = Tensor(3.0, requires_grad=True)

# Perform operations
c = a * b
d = c + a

# Compute gradients
d.backward()

# Access gradients
print(f"Gradient of a: {a.grad}")
print(f"Gradient of b: {b.grad}")
```

## Contributing
Contributions are welcome! If you'd like to contribute, please fork the repository and submit a pull request.
