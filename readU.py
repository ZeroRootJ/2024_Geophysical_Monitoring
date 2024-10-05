import pickle
import numpy as np
import matplotlib.pyplot as plt
from Utils import IdxConv

nx = 400
nz = 200
ic = IdxConv(nx)

U = pickle.load(open("U", "rb"))

n = 700

wavefield_n = np.zeros((nx, nz))
for index in range(nx * nz):
    x, z = ic.unflatten_index(index)
    wavefield_n[x, z] = U[n][index]

plt.figure(figsize=(10, 10))
plt.imshow(wavefield_n.T, cmap='gray')
plt.title(f"Snapshot at Timestep {n}")
c = plt.colorbar()
plt.show()