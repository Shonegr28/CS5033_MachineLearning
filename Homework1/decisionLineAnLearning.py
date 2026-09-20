import matplotlib.pyplot as plt

import numpy as np

x1 = np.linspace(-1, 1, 400)

x2_abc = (0.5 - 1.1 * x1) / 0.7
x2_d = (-0.5 - 1.1 * x1) / 0.7

plt.xlim(-1, 1)
plt.ylim(-1.5, 1.5)

plt.plot(x1, x2_abc, color="black", linewidth=2, label="After (a), (b), (c)")
plt.plot(x1, x2_d, color="red", linewidth=2, label="After (d)")

class0_x = [-0.1, 0.2]
class0_y = [-1.0, -0.9]

class1_x = [0.6, 0.0]
class1_y = [0.8, 0.0]

plt.scatter(class0_x, class0_y, s=70, label="Class 0")
plt.scatter(class1_x, class1_y, s=70, label="Class 1")

plt.text(-0.07, -0.95, "(a)")
plt.text(0.23, -0.85, "(b)")

plt.text(0.63, 0.85, "(c)")
plt.text(0.03, 0.05, "(d)")

plt.axhline(y=0, color="black", linewidth=2)
plt.axvline(x=0, color="black", linewidth=2)
plt.xlabel("x1")
plt.ylabel("x2")

plt.legend()

plt.show()