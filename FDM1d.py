# import module
import numpy as np
import matplotlib.pyplot as plt

# Declare Parameters

dx = 0.005;
nx = 200
vmax = 2;
tmax = 1.0
dt = 0.001;
nt = 1000
fmax = 20


# define source
# Ricker or fdgaus

def ricker_wavelet(T, dt, peak_frequency):
    t = np.arange(0, T, dt)
    t_peak = 0.15 * T / 2.0
    alpha = (np.pi * peak_frequency * (t - t_peak)) ** 2
    return (1 - 2 * alpha) * np.exp(-alpha)


def fdgaus_wavelet(T, dt):
    a = 10  # Amplitude
    b = 50  # Middle Point
    c = 10  # Frequency
    t = np.arange(0, T)
    print(len(t))
    return a * np.exp(-(t - b) ** 2 / (2 * c ** 2)) * (-2 * (t - b) / (2 * c ** 2))


source = ricker_wavelet(tmax, dt, fmax)
# source = fdgaus_wavelet(nt, dt)
plt.plot(source)
# plt.plot(source2)
# plt.xlabel("Time Step")
plt.xticks([int(nt // 10) * i for i in range(11)], [int(1000 * dt * nt // 10) * i for i in range(11)])
plt.xlabel("Time (ms)")
plt.ylabel("Amplitude")

plt.show()
# Create a 2D wave field.
# The 2D array consists of a time axis and a 1D space axis.
U = np.zeros((nt, nx))

##**1차원 모델에선 전체 파동장을 U라는 2차원 array로 지정했지만, 2차원 모델에서는 3D로 지정해야 합니다.
##2차원 모델 생성시 삼중중첩 for문에서 3차원 array U를 활용한 계산은 비효율적입니다.
##임시로 사용할 특정 시간대의 2차원 array를 지정하고, 3차원 array에 저장하는 방식으로 진행하는 것이 효율적입니다.

# Create a shource location
f = np.zeros(nx)
f[int(nx / 2)] = 1

# Create a source wavelet
time = np.linspace(0 * dt, nt * dt, nt)
source = ricker_wavelet(tmax, dt, fmax)

# Initiate
U[0][:] = 0
U[1][:] = 0

# Iteration
for j in range(nt - 2):
    for i in range(1, nx - 1):
        # Boundary condition
        #         U[j+2][0] = 0
        #         U[j+2][nx-1] = 0
        # Reynolds Absorbing Boundary condition
        U[j + 2][0] = vmax * dt / dx * (U[j + 1][1] - U[j + 1][0]) + U[j + 1][0]
        U[j + 2][nx - 1] = vmax * dt / dx * (U[j + 1][nx - 2] - U[j + 1][nx - 1]) + U[j + 1][nx - 1]
        # Wave equation
        U[j + 2][i] = vmax ** 2 * dt ** 2 * (
                    (U[j + 1][i + 1] - 2 * U[j + 1][i] + U[j + 1][i - 1]) / dx ** 2 + source[j] * f[i]) + 2 * U[j + 1][
                          i] - U[j][i]

# Plot
plt.figure(figsize=(10, 10))
plt.imshow(U, aspect='auto', cmap='gray')
plt.title("1D FDM Result")
# Converting Unit - Distance Step to Meter
plt.xticks([int(nx // 10) * i for i in range(11)], [int(1000 * dx * nx // 10) * i for i in range(11)])
plt.xlabel("distance (m)")
# Converting Unit - Time Step to Microsecond
plt.yticks([int(nt // 10) * i for i in range(11)], [int(1000 * dt * nt // 10) * i for i in range(11)])
plt.ylabel("Time (ms)")
c = plt.colorbar()
plt.show()
