import numpy as np
import matplotlib.pyplot as plt


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


X = np.array([[0, 0, 1],
              [0, 1, 1],
              [1, 0, 1],
              [1, 1, 1]])

D = np.array([[0], [1], [0], [1]])

W_B = np.random.rand(1, 3)
W_SGD = W_B
dw_sum = np.zeros((3, 1))
eta = 0.6
E1 = []
E2 = []
# Batch
for i in range(1000+1):
    temp_e = []
    for j in range(D.shape[0]):
        x = np.reshape(X[j], (3, 1))
        d = np.array([D[j][0]])
        v = np.dot(W_B, x)
        y = sigmoid(v)
        e = d - y
        if i % 10 == 0:
            temp_e.append(-1 * e[0][0])
        delta = y * (1 - y) * e
        dw = eta * delta * x
        dw_sum = dw_sum + dw
    if i % 10 == 0:
        E1.append((temp_e[0] + temp_e[1] + temp_e[2] + temp_e[3])/4)
    wd_avg = np.divide(dw_sum, 4)
    W_B = W_B + np.transpose(wd_avg)

# SGD

for i in range(1000+1):
    temp_e = []
    for j in range(D.shape[0]):
        x = np.reshape(X[j],(3,1))
        d = D[j,:]
        v = np.dot(W_SGD, x)
        y = sigmoid(v)
        e = d - y
        if i % 10 == 0:
            temp_e.append(-1 * e[0][0])
        delta = y*(1-y)*e
        dw = eta * delta * x
        W_SGD = W_SGD + np.transpose(dw)
    if i % 10 == 0:
        E2.append((temp_e[0] + temp_e[1] + temp_e[2] + temp_e[3])/4)

plt.plot(E2, 'r')
plt.plot(E1, 'b')
plt.xlabel('Epochs')
plt.ylabel('Error')
plt.show()