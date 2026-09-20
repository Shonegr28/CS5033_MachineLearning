import matplotlib.pyplot as plt
import numpy as np

x1 = np.linspace(-2, 2, 400)

x2 = (0.5 + x1) / 0.3

plt.xlim(-2, 2)
plt.ylim(-5, 8)

plt.plot(x1, x2, linewidth=2)

plt.fill_between(x1, -5, x2, alpha=0.25, label="Class 1")
plt.fill_between(x1, x2, 8, alpha=0.25, label="Class 0")

plt.scatter(-0.5, 0, s=60)
plt.scatter(0, 1.67, s=60)

plt.text(-0.9, -0.7, "(-0.5, 0)")
plt.text(0.1, 1.8, "(0, 1.67)")

plt.axhline(y=0, color="black", linewidth=2)
plt.axvline(x=0, color="black", linewidth=2)



plt.xlabel("x1")
plt.ylabel("x2")

plt.legend()

plt.show()