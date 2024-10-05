import numpy as np
import pickle
import matplotlib.pyplot as plt
from Utils import IdxConv

# Declare Parameters
dx = 0.005
dz = 0.005
nx = 400
nz = 200
vmax = 2
tmax = 1.0
dt = 0.001
nt = 1000
fmax = 20


# Define source wavelet
def ricker_wavelet(T, dt, peak_frequency):
    t = np.arange(0, T, dt)
    t_peak = 0.15 * T / 2.0
    alpha = (np.pi * peak_frequency * (t - t_peak)) ** 2
    return (1 - 2 * alpha) * np.exp(-alpha)


# Initialize the index converter with the value of nx
idx_conv = IdxConv(nx)

# Source wavelet
source = ricker_wavelet(tmax, dt, fmax)
# plt.plot(source)
# plt.show()

# Create a 2D array U for wave field: time x (flattened location)
U = np.zeros((nt, nx * nz))

# Place source at z=0 (ground) and in the middle of x-axis (x=nx/2)
f = np.zeros(nx * nz)
f[idx_conv.flatten_index(int(nx / 2), 0)] = 1  # Source at ground, middle of x-axis

# Initialize U for first two time steps
U[0][:] = 0
U[1][:] = 0

# Iterate over time and space
for j in range(nt - 2):
    if j%10 == 0:
        print(f"Time step {j} / {nt}")

    for index in range(nx*nz):
        # Convert the 1D index back to (x, z)
        x, z = idx_conv.unflatten_index(index)

        # Source Initialization
        if x == int(nx / 2) and z == 1:
            f = source[j]
        else:
            f = 0

        # Left boundary condition: x = 0
        if x == 0:
            U[j + 2][index] = (
                    U[j + 1][index] + U[j + 1][idx_conv.flatten_index(1, z)] - U[j][idx_conv.flatten_index(1, z)] +
                    vmax * dt / dx * (
                            (U[j + 1][idx_conv.flatten_index(1, z)] - U[j + 1][idx_conv.flatten_index(0, z)]) -
                            (U[j][idx_conv.flatten_index(2, z)] - U[j][idx_conv.flatten_index(1, z)])
                    )
            )

        # Right boundary condition: x = nx - 1
        elif x == nx - 1:
            U[j + 2][index] = (
                    U[j + 1][index] + U[j + 1][idx_conv.flatten_index(nx - 2, z)] - U[j][
                idx_conv.flatten_index(nx - 2, z)] -
                    vmax * dt / dx * (
                            (U[j + 1][idx_conv.flatten_index(nx - 1, z)] - U[j + 1][
                                idx_conv.flatten_index(nx - 2, z)]) -
                            (U[j][idx_conv.flatten_index(nx - 2, z)] - U[j][idx_conv.flatten_index(nx - 3, z)])
                    )
            )

        # Bottom boundary condition: z = nz - 1
        elif z == nz - 1:
            U[j + 2][index] = (
                    U[j + 1][index] + U[j + 1][idx_conv.flatten_index(x, nz - 2)] - U[j][
                idx_conv.flatten_index(x, nz - 2)] -
                    vmax * dt / dz * (
                            (U[j + 1][idx_conv.flatten_index(x, nz - 1)] - U[j + 1][
                                idx_conv.flatten_index(x, nz - 2)]) -
                            (U[j][idx_conv.flatten_index(x, nz - 2)] - U[j][idx_conv.flatten_index(x, nz - 3)])
                    )
            )

        # Interior points
        elif 0 < x < nx - 1 and 0 < z  < nz - 1:
            # 2D wave equation (central difference for both spatial dimensions)
            U[j + 2][index] = (

                    2 * U[j + 1][index]

                    - U[j][index]

                    + (vmax ** 2) * (dt ** 2) * (
                            (U[j + 1][idx_conv.flatten_index(x + 1, z)]
                             - 2 * U[j + 1][idx_conv.flatten_index(x, z)]
                             + U[j + 1][idx_conv.flatten_index(x - 1, z)]) / (dx ** 2)
                            )

                    + (vmax ** 2) * (dt ** 2) * (
                            (U[j + 1][idx_conv.flatten_index(x, z + 1)]
                             - 2 * U[j + 1][idx_conv.flatten_index(x, z)]
                             + U[j + 1][idx_conv.flatten_index(x, z - 1)]) / (dz ** 2)
                    )

                    + (vmax ** 2) * (dt ** 2) * f
            )

# Set the desired timestep (n)
n = 750
# Change 'n' to the desired timestep

# Plot the wave field at timestep n
wavefield_n = np.zeros((nx, nz))
for index in range(nx * nz):
    x, z = idx_conv.unflatten_index(index)
    wavefield_n[x, z] = U[n][index]

# print(U[50])
#
#
plt.figure(figsize=(10, 10))
plt.imshow(wavefield_n.T, cmap='gray')
plt.title(f"Snapshot at Timestep {n}")
c = plt.colorbar()
plt.show()

pickle.dump(U, open('U', 'wb'))



