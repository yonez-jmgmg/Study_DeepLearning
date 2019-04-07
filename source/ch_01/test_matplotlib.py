import numpy as np
import matplotlib.pyplot as plt

x = np.arange(0, 10, 0.1)
# print(x)

y = np.sin(x)
# print(y)

plt.plot(x, y, 'o-')
plt.show()
