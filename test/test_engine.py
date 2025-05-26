import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from nanoautograd.tensor import Tensor

def test_case():
    x = Tensor(-4.0)
    x.requires_grad = True
    z = x * 2 + 2 + x
    q = z.relu() + z * x
    h = (z * z).relu()
    y = h + q + q * x
    y.backward()
    xmg, ymg = x, y
    print("nanoautograd: ")
    print(f"Forward pass: {ymg.data}, backward_pass: {xmg.grad}")

    x = torch.Tensor([-4.0]).double()
    x.requires_grad = True
    z = x * 2 + 2 + x
    q = z.relu() + z * x
    h = (z * z).relu()
    y = h + q + q * x
    y.backward()
    xpt, ypt = x, y

    print("pytorch: ")
    print(f"Forward pass: {ypt.data.item()}, backward_pass: {xpt.grad.item()}")

test_case()