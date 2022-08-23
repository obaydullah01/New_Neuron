import numpy as np
import matplotlib.pyplot as plt

x = np.array([[0, 0, 1, 1],
              [0, 1, 0, 1],
              [-1, -1, -1, -1]])

yd = np.array([0, 0, 0, 1])
w = np.random.rand(1, x.shape[0])

x_d = np.array([-1, 0, 1, 2])
dc_b = []
slop = -1 * w[0][0] / w[0][1]
y_d = []
for k in x_d:
    y_d.append(slop * k + w[0][2] / w[0][1])
dc_b.append(y_d)
e_list = []
w_list = []
eta = 0.1

for i in range(1000 + 1):
    v_k = np.dot(w, x)
    y = 1 / (1 + np.exp(-v_k))
    e = yd - y
    if i % 100 == 0:
        print(f"[{i} - {e}]")
    e_list.append(np.sum(np.square(np.subtract(np.mean(e[0]), e[0]))) / 4)

    del_w = eta * np.dot(e, np.transpose(x))
    w = w + del_w
    w_list.append(w[0])
    if i != 0 and i % 100 == 0:
        slop = -1 * w[0][0] / w[0][1]
        y_d = []
        for j in x_d:
            y_d.append(slop * j + w[0][2] / w[0][1])
        dc_b.append(y_d)
# print(dc_b)
# if i != 0 and i % 100==0:
#     slop = -1 * w_list[i][0] / w_list[i][1]
#     y_d = []
#     for j in x_d:
#         y_d.append(slop * j + w_list[i][2] / w_list[i][1])
#     dc_b.append(y_d)

fig, ax = plt.subplots()
fig.suptitle('decision boundary')
ax.plot([0, 0, 1], [0, 1, 0], 'bo')
ax.plot([1], [1], 'ro')
n = 0
for t in dc_b:
    ax.plot(x_d, t, label=f"{n}th")
    n += 1
ax.legend()

fig1, ax1 = plt.subplots()
fig1.suptitle('error')
# x_label = [i for i in range(1000+1)]
x_label = []
for i in range(1001):
    x_label.append(i)
ax1.plot(x_label, e_list)

for i in range(0, 1000, 100):
    print(f"epochs {i} -> {e_list[i]}\n")
plt.show()
