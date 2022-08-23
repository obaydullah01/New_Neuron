import numpy as np
import matplotlib.pyplot as plt


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


input1 = np.array([[0, 0, 1],
                   [0, 1, 1],
                   [1, 0, 1],
                   [1, 1, 1]])

target = np.array([[0], [0], [1], [1]])
W = np.random.rand(1, 3)
eta = 0.9
E1 = []
for i in range(10000+1):
    temp_e = []
    for j in range(target.shape[0]):
        x = np.reshape(input1[j], (3, 1))
        d = target[j, :]
        v = np.dot(W, x)
        y = sigmoid(v)
        e = (d - y)

        if i % 10 == 0:
            temp_e.append(-1*e[0][0])
        delta = y*(1-y)*e
        dw = eta * delta * x
        W = W + np.transpose(dw)
    if i % 10 == 0:
        E1.append((temp_e[0] + temp_e[1] + temp_e[2] + temp_e[3])/4)
# x1 = int(input())
# x2 = int(input())
# b = 1
# v1 = (W[0][0] * x1 + W[0][1] * x2 + W[0][2] * b)
# print(round(sigmoid(v1)))

for k in range(4):
    x = np.reshape(input1[k], (3, 1))
    v = np.dot(W, x)
    y = sigmoid(v)
    print(np.round(y[0][0]))

# fig, ax = plt.subplots()
# fig.suptitle('error')
# x_ax = []
# for i in range(100+1):
#     x_ax.append(i)
# ax.plot(x_ax, e_list)
# plt.show()

plt.plot(E1)
plt.xlabel('Epoch')
plt.ylabel('Error')
plt.show()


