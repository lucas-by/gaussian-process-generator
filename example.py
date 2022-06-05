import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl

from gaussian_process import GaussianProcess as gp


n = 200
xmin = -5
xmax = 5
hyperparam = 0.3

x = np.linspace(xmin, xmax, n)
mpl.style.use("seaborn")

plt.figure(figsize=(16, 4))

ax1 = plt.subplot(1, 3, 1)
ax1.set_title("Gaussian kernel", fontsize=15)

ax2 = plt.subplot(1, 3, 2)
ax2.set_title("Brownian kernel", fontsize=15)

ax3 = plt.subplot(1, 3, 3)
ax3.set_title("Symmetric kernel", fontsize=15)

for idx in range(3):
    Y = gp.build_randfunc(x, hyperparam, gp.kernelGA)
    ax1.plot(x, Y)

    Y = gp.build_randfunc(x, hyperparam, gp.kernelBR)
    ax2.plot(x, Y)

    hyperparam = hyperparam*4
    Y = gp.build_randfunc(x, hyperparam, gp.kernelSY)
    ax3.plot(x, Y)
    hyperparam = hyperparam/4

ax1.autoscale(enable=True, axis='x', tight=True)
ax2.autoscale(enable=True, axis='x', tight=True)
ax3.autoscale(enable=True, axis='x', tight=True)

plt.tight_layout()
plt.show()
