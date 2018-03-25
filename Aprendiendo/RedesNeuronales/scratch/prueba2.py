import numpy as np
import matplotlib.pyplot as plt

pi = np.pi
e = np.exp(1)
t = np.linspace(-pi,pi,3600)

x = np.sin(t)
y = e**-(2*t)

# y = np.sin(t)

plt.plot(x,y)
plt.show()
