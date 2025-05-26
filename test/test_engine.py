import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from nanoautograd.tensor import Tensor

def test_case1():
    # nanoautograd
    x = Tensor(-4.0)
    x.requires_grad = True
    z = x * 2 + 2 + x
    q = z.relu() + z * x
    h = (z * z).relu()
    y = h + q + q * x
    y.backward()
    print("nanoautograd:")
    print(f"Forward pass: {y.data}, backward_pass: {x.grad}")

    # PyTorch
    x = torch.tensor([-4.0], dtype=torch.float64, requires_grad=True)
    z = x * 2 + 2 + x
    q = z.relu() + z * x
    h = (z * z).relu()
    y = h + q + q * x
    y.backward()
    print("pytorch:")
    print(f"Forward pass: {y.item()}, backward_pass: {x.grad.item()}")

def test_case2():
    # nanoautograd
    x = Tensor(2.0)
    x.requires_grad = True
    z = (x + 1).log()
    q = z * x + x ** 3
    y = q + z
    y.backward()
    print("nanoautograd:")
    print(f"Forward pass: {y.data}, backward_pass: {x.grad}")

    # PyTorch
    x = torch.tensor([2.0], dtype=torch.float64, requires_grad=True)
    z = (x + 1).log()
    q = z * x + x ** 3
    y = q + z
    y.backward()
    print("pytorch:")
    print(f"Forward pass: {y.item()}, backward_pass: {x.grad.item()}")

def test_case3():
    # nanoautograd
    x = Tensor(1.5)
    x.requires_grad = True
    z = x.tanh() + x * x
    y = z + x * z
    y.backward()
    print("nanoautograd:")
    print(f"Forward pass: {y.data}, backward_pass: {x.grad}")

    # PyTorch
    x = torch.tensor([1.5], dtype=torch.float64, requires_grad=True)
    z = x.tanh() + x * x
    y = z + x * z
    y.backward()
    print("pytorch:")
    print(f"Forward pass: {y.item()}, backward_pass: {x.grad.item()}")

def test_case4():
    # nanoautograd
    x = Tensor(-1.0)
    x.requires_grad = True
    z = (x * 3 + 4).relu()
    q = z ** 2 + x
    y = q * z + z
    y.backward()
    print("nanoautograd:")
    print(f"Forward pass: {y.data}, backward_pass: {x.grad}")

    # PyTorch
    x = torch.tensor([-1.0], dtype=torch.float64, requires_grad=True)
    z = (x * 3 + 4).relu()
    q = z ** 2 + x
    y = q * z + z
    y.backward()
    print("pytorch:")
    print(f"Forward pass: {y.item()}, backward_pass: {x.grad.item()}")

if __name__ == "__main__":
    test_case1()
    test_case2()
    test_case3()
    test_case4()