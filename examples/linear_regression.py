# Example of a simple neural network using autograd engine.
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nanoautograd import Tensor
from nanoautograd.nn import Linear, MSELoss
from nanoautograd.optim import SGD, Adam
import numpy as np

def count_trainable_params(model):
    return sum(p.data.size for p in model.parameters())

def main():
    n_samples = 10
    # 2D data: (x1, x2)
    X = np.random.randn(n_samples, 1) * 10
    
    # y = (2*x+1) + some noise 
    y_inter = 2*X + 1
    y = y_inter + np.random.randn(n_samples, 1) * 0.5

    X_tensor = Tensor(X) 
    y_tensor = Tensor(y) 

    model = Linear(1, 1, bias=True)
    loss_fn = MSELoss()
    optimizer = SGD(model.parameters(), learning_rate=0.001)
    print("No. of trainable parameters: ", count_trainable_params(model))

    epochs = 100
    print("Training linear regression model...")
    print(f"Initial parameters: weight={model.parameters()[0].data}, bias={model.bias.data[0]}")
    
    for epoch in range(epochs):
        # Forward pass
        y_pred = model(X_tensor)
        loss = loss_fn(y_pred, y_tensor)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Print progress
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.data:.4f}")
    
    print(f"Final parameters: weight={model.parameters()[0].data}, bias={model.bias.data[0]}")
    print(f"True parameters: weight=2.0000, bias=1.0000")

if __name__ == "__main__":
    main()