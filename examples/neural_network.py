# Example of a simple neural network using autograd engine.
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nanoautograd import Tensor
from nanoautograd.nn import Linear, Sequential, ReLU, MSELoss
from nanoautograd.optim import Adam, SGD
import numpy as np

def count_trainable_params(model):
    return sum(p.data.size for p in model.parameters())

def main():
    n_samples = 5
    # 2D data: (x1, x2)
    X = np.random.randn(n_samples, 2) # (5,2)
    
    # y = (x1^2 + x2^2) + some noise 
    y_inter = (X[:,0]**2 + X[:,1]**2).reshape(-1,1)
    y = y_inter + np.random.randn(n_samples, 1) * 0.1

    X_mean, X_std = X.mean(axis=0), X.std(axis=0)
    y_mean, y_std = y.mean(), y.std()

    X_normalized = (X - X_mean)/X_std
    y_normalized = (y - y_mean)/y_std

    X_tensor = Tensor(X_normalized) 
    y_tensor = Tensor(y_normalized) 

    model = Sequential(
        Linear(2,16,bias=False),
        ReLU(),
        Linear(16,8,bias=False),
        ReLU(),
        Linear(8,1,bias=False)
    )

    loss_fn = MSELoss()
    optimizer = Adam(model.parameters(), learning_rate=0.01)
    epochs = 100
    print("No. of trainable parameters: ", count_trainable_params(model))

    # Training loop
    for epoch in range(epochs):
        y_pred = model(X_tensor)
        loss = loss_fn(y_pred,y_tensor)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            print(f"Epoch [{epoch + 1:02d}/{epochs}] | Loss: {loss.data:.4f}")

    test_X = np.array([
        [1.0, 1.0],
        [2.0, 0.0],
        [0.0, 2.0],
        [-1.0, -1.0]
    ])
    
    test_X_normalized = (test_X - X_mean) / X_std
    test_X_tensor = Tensor(test_X_normalized)
    
    with np.printoptions(precision=4, suppress=True):
        predictions_normalized = model(test_X_tensor).data
        predictions = predictions_normalized * y_std + y_mean
        expected = (test_X[:, 0] ** 2 + test_X[:, 1] ** 2).reshape(-1, 1)
        
        print("\nTest Results:")
        print("Point\t\t\tPrediction\tExpected")
        for i in range(len(test_X)):
            point = f"({test_X[i, 0]:.1f}, {test_X[i, 1]:.1f})"
            print(f"{point}\t\t{predictions[i, 0]:.4f}\t\t{expected[i, 0]:.4f}")

if __name__ == "__main__":
    main()