import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

data = pd.read_csv('./data/train.csv')

data = np.array(data)
m, n = data.shape
np.random.shuffle(data)

data_dev = data[:1000].T
Y_dev = data_dev[0]
X_dev = data_dev[1:n]
X_dev = X_dev / 255.

data_train = data[1000:m].T
Y_train = data_train[0]
X_train = data_train[1:n]
X_train = X_train / 255.
_, m_train = X_train.shape

def ReLU(Z):
    # ReLU function
    # Return the maximum of 0 and Z
    # In other words:
    # If Z is positive, return Z
    # If Z is negative, return 0
    return np.maximum(0, Z)

def ReLU_deriv(Z):
    # Derivative of ReLU function
    # Return 1 if Z is positive, 0 otherwise
    return Z > 0

def softmax(Z):
    # Softmax function
    # Return the exponential of Z, divided by the sum of the exponential of Z
    return np.exp(Z) / sum(np.exp(Z))

def one_hot(Y):
    # One-hot encoding function
    # Return a matrix of 0s with the same shape of Y
    # Set the value of the index of Y to 1
    one_hot_Y = np.zeros((Y.size, Y.max() + 1))
    one_hot_Y[np.arange(Y.size), Y] = 1
    # Return the transposed matrix to match the shape of the input
    return one_hot_Y.T

def init_params():
    # Initialize the parameters
    W1 = np.random.rand(10, 784) - 0.5
    b1 = np.random.rand(10, 1) - 0.5
    
    W2 = np.random.rand(10, 10) - 0.5
    b2 = np.random.rand(10, 1) - 0.5
    
    W3 = np.random.rand(10, 10) - 0.5
    b3 = np.random.rand(10, 1) - 0.5
    
    return W1, b1, W2, b2, W3, b3

def forward_prop(W1, b1, W2, b2, W3, b3, X):
    Z1 = W1.dot(X) + b1
    A1 = ReLU(Z1)
    
    Z2 = W2.dot(A1) + b2
    A2 = ReLU(Z2)
    
    Z3 = W3.dot(A2) + b3
    A3 = softmax(Z3)
    
    return Z1, A1, Z2, A2, Z3, A3
    
def back_prop(Z1, A1, Z2, A2, Z3, A3, W2, W3, X, Y):
    one_hot_Y = one_hot(Y)
    
    dZ3 = A3 - one_hot_Y
    dW3 = 1 / m * dZ3.dot(A2.T)
    db3 = 1 / m * np.sum(dZ3)
    
    dZ2 = W3.T.dot(dZ3) * ReLU_deriv(Z2)
    dW2 = 1 / m * dZ2.dot(A1.T)
    db2 = 1 / m * np.sum(dZ2)
    
    dZ1 = W2.T.dot(dZ2) * ReLU_deriv(Z1)
    dW1 = 1 / m * dZ1.dot(X.T)
    db1 = 1 / m * np.sum(dZ1)
    
    return dW1, db1, dW2, db2, dW3, db3

def adam(params, grads, v, s, t, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
    for param, grad, v_param, s_param in zip(params, grads, v, s):
        v_param = beta1 * v_param + (1 - beta1) * grad
        s_param = beta2 * s_param + (1 - beta2) * (grad ** 2)
        v_corrected = v_param / (1 - beta1 ** t)
        s_corrected = s_param / (1 - beta2 ** t)
        param -= learning_rate * v_corrected / (np.sqrt(s_corrected) + epsilon)
    return params, v, s

def gradient_descent(X, Y, iters, learning_rate):
    W1, b1, W2, b2, W3, b3 = init_params()
    v = [np.zeros_like(W1), np.zeros_like(b1), np.zeros_like(W2), np.zeros_like(b2), np.zeros_like(W3), np.zeros_like(b3)]
    s = [np.zeros_like(W1), np.zeros_like(b1), np.zeros_like(W2), np.zeros_like(b2), np.zeros_like(W3), np.zeros_like(b3)]
    params = [W1, b1, W2, b2, W3, b3]
    
    for i in range(iters):
        Z1, A1, Z2, A2, Z3, A3 = forward_prop(W1, b1, W2, b2, W3, b3, X)
        dW1, db1, dW2, db2, dW3, db3 = back_prop(Z1, A1, Z2, A2, Z3, A3, W2, W3, X, Y)
        grads = [dW1, db1, dW2, db2, dW3, db3]
        
        params, v, s = adam(params, grads, v, s, i+1, learning_rate)
        W1, b1, W2, b2, W3, b3 = params
        
        if i % 10 == 0:
            print(f"Iteration {i}")
            predictions = get_predictions(A3)
            print(f"Accuracy: {get_accuracy(predictions, Y)}")
    
    return W1, b1, W2, b2, W3, b3

def get_predictions(A3):
    # Get the index of the maximum value in A3
    return np.argmax(A3, 0)

def get_accuracy(predictions, Y):
    # Calculate the accuracy of the predictions
    print(predictions, Y)
    # Return the number of correct predictions
    return np.sum(predictions == Y) / Y.size

def make_predictions(X, W1, b1, W2, b2, W3, b3):
    _, _, _, _, _, A3 = forward_prop(W1, b1, W2, b2, W3, b3, X)
    predictions = get_predictions(A3)
    return predictions

def test_prediction(index, W1, b1, W2, b2, W3, b3):
    current_image = X_train[:, index, None]
    prediction = make_predictions(X_train[:, index, None], W1, b1, W2, b2, W3, b3)
    label = Y_train[index]
    print("Prediction: ", prediction)
    print("Label: ", label)
    
    current_image = current_image.reshape((28, 28)) * 255
    plt.gray()
    plt.imshow(current_image, interpolation='nearest')
    plt.show()
    

W1, b1, W2, b2, W3, b3 = gradient_descent(X_train, Y_train, 1000, 0.001)  # Note: typically use a lower learning rate with Adam

test_prediction(3, W1, b1, W2, b2, W3, b3)