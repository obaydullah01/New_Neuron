import numpy as np


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


X = np.array([[0, 0, 1],
              [0, 1, 1],
              [1, 0, 1],
              [1, 1, 1]])

D = np.array([[0, 1, 1, 0]])
W1 = np.random.rand(4, 3)
W2 = np.random.rand(1, 4)

eta = 0.9

for i in range(10000):
    for j in range(D.shape[1]):
        x = np.reshape(X[j], (3, 1))
        d = D[0][j]
        v1 = np.dot(W1, x)
        y1 = sigmoid(v1)
        v = np.dot(W2, y1)
        y = sigmoid(v)
        e = d - y

        delta = y * (1 - y) * e
        e1 = np.dot(W2.transpose(), delta)
        delta1 = y1 * (1 - y1) * e1

        dw1 = eta * np.dot(delta1, np.transpose(x))
        W1 = W1 + dw1
        dw2 = eta * np.dot(delta, np.transpose(y1))
        W2 = W2 + dw2

for i in range(D.shape[1]):
    x = np.reshape(X[i], (3, 1))
    v1 = np.dot(W1, x)
    y1 = sigmoid(v1)
    v = np.dot(W2, y1)
    y = sigmoid(v)
    print(np.round(y))

