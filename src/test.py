import numpy as np

flow_rate = np.arange(1, 9, 1)
k = np.arange(0.1, 0.5, 0.05)[::-1]

i = np.arange(1, 9, 1)

for j in i:
    print(j, 1/j)


