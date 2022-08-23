import numpy as np
import matplotlib.pyplot as plt


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


X = np.array([[0, 0, 1],
              [0, 1, 1],
              [1, 0, 1],
              [1, 1, 1]])

D = np.array([[0], [0], [1], [1]])

W = np.random.rand(1, 3)
dw_sum = np.zeros((3, 1))
eta = 0.6
E1 = []
for i in range(1000):
    temp_e = []
    for j in range(D.shape[0]):
        x = np.reshape(X[j], (3, 1))
        d = np.array([D[j][0]])
        v = np.dot(W, x)
        y = sigmoid(v)
        e = d - y
        if i % 10 == 0:
            temp_e.append(-1 * e[0][0])
        delta = y * (1 - y) * e
        dw = eta * delta * x
        dw_sum = dw_sum + dw
    if i % 10 == 0:
        E1.append((temp_e[0] + temp_e[1] + temp_e[2] + temp_e[3]) / 4)
    wd_avg = np.divide(dw_sum, 4)
    W = W + np.transpose(wd_avg)

plt.plot(E1)
plt.xlabel('Epoch')
plt.ylabel('Error')
plt.show()

while 1:
    x1 = int(input())
    x2 = int(input())
    b = 1
    v = (W[0][0] * x1 + W[0][1] * x2 + W[0][2] * b)
    y = sigmoid(v)
    print(np.round(y))

