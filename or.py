import numpy as np
import matplotlib.pyplot as plt

x = np.array([[0, 0, 1, 1],
              [0, 1, 0, 1],
              [-1, -1, -1, -1]])

yd = np.array([[0, 1, 1, 1]])
w = np.random.rand(1, x.shape[0])

x_d = np.array([-1, 0, 1, 2])
dc_b = []
slop = 1*w[0][0]/w[0][1]
y_d = []
for k in x_d:
    y_d.append(slop*k+w[0][2]/w[0][1])
dc_b.append(y_d)
e_list = []
w_list = []

for i in range(1000+1):
    v_k = np.dot(w, x)
    y = 1/(1+np.exp(-v_k))
    e = yd - y
    e_list.append(np.sum(np.square(np.subtract(np.mean(e[0]), e[0]))) / 4)
    eta = 0.5
    del_w = eta*np.dot(e, np.transpose(x))
    w = w + del_w
    w_list.append(w[0])
    if i!=0 and i%100==0:
        slop = -1 * w[0][0] / w[0][1]
        y_d = []
        for j in x_d:
            y_d.append(slop * j + w[0][2] / w[0][1])
        dc_b.append(y_d)


fig, ax = plt.subplots()
fig.suptitle('decision boundary')
ax.plot([0, 1, 1], [1, 0, 1],'bo')
ax.plot([0], [0], 'r+')
for t in dc_b:
    ax.plot(x_d,t)
ax.legend()

fig1, ax1 = plt.subplots()
fig1.suptitle('error')
x_label = [i for i in range(1000+1)]
ax1.plot(x_label, e_list)
ax1.legend()
plt.show()




