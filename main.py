from CubatureKalman import *
from SimulateSystems import *
import numpy as np
import matplotlib.pyplot as plt
def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())

rmse_test = np.zeros(4)
M = 1
for i in range(0, M):
    N = 100
    noise = 1e-4

    x0 = np.array([0.0, 0.0, 0.0])
    P = np.array([[1e-4, 0.0, 0.0],
                  [0.0, 1e-4, 0.0],
                  [0.0, 0.0, 1e-4]])
    x = np.random.multivariate_normal(x0, P)
    Q = np.array([[noise, 0.0, 0.0],
                  [0.0, noise, 0.0],
                  [0.0, 0.0, noise]])
    R = np.array([[noise]])
    h = np.array([[1.0, 0.0, 0.0]])

    data = StochasticNonlinearSystem(x, N, 4, 0.5)
    measurement, state = data.simulate(Q, R)
    measurements = measurement.copy()
    true_measurement, true_state = data.simulate()
    outlier_indices = np.random.randint(0, N, 25)
    outliers = np.random.normal(0, R[0] * 10000, len(outlier_indices))
    measurement_outliers = measurement.copy()
    measurement_outliers[outlier_indices] = outliers

    ckf = CubatureKalmanFilter(x, P, Q, R, data.fx, data.hx, h)
    ackf = AdaptiveCubatureKalmanFilter(x, P, Q, R, data.fx, data.hx, h, 0.3)
    rckf = RobustCubatureKalmanFilter(x, P, Q, R, data.fx, data.hx, h, 1)
    hckf = HybridCubatureKalmanFilter(x, P, Q, R, data.fx, data.hx, h, 0.3, 1)

    ckf_state, ckf_measurement = ckf.filter(measurement)
    rckf_state, rckf_measurement = rckf.filter(measurement)
    ackf_state, ackf_measurement = ackf.filter(measurements)
    hckf_state, hckf_measurement = hckf.filter(measurement)

    rmse_test[0] += rmse(true_measurement, ckf_measurement)
    rmse_test[1] += rmse(true_measurement, ackf_measurement)
    rmse_test[2] += rmse(true_measurement, rckf_measurement)
    rmse_test[3] += rmse(true_measurement, hckf_measurement)
rmse_test /= M
plt.subplot(4, 1, 1)
plt.plot(true_measurement, 'red', label='true')
plt.plot(measurement_outliers,'black', label='measurement')
plt.plot(ckf_state[:, 0], 'blue', label='ckf')
plt.title('Кубатурный фильтр')
plt.ylim(-1, 1)
plt.grid()
plt.legend()
plt.subplot(4, 1, 2)
plt.plot(true_measurement, 'red', label='true')
plt.plot(measurement_outliers,'black', label='measurement')
plt.plot(ackf_state[:, 0], 'green', label='ackf')
plt.title('Адаптивный фильтр')
plt.ylim(-1, 1)
plt.grid()
plt.legend()
plt.subplot(4, 1, 3)
plt.plot(true_measurement, 'red', label='true')
plt.plot(measurement_outliers,'black', label='measurement')
plt.plot(rckf_state[:, 0], 'orange', label='rckf')
plt.title('Робастный фильтр')
plt.ylim(-1, 1)
plt.grid()
plt.legend()
plt.subplot(4, 1, 4)
plt.plot(true_measurement, 'red', label='true')
plt.plot(measurement_outliers,'black', label='measurement')
plt.plot(hckf_state[:, 0], 'purple', label='ckf')
plt.title('Гибридный фильтр')
plt.ylim(-1, 1)
plt.grid()
plt.legend()
plt.subplots_adjust(hspace=0.5)
plt.show()
print(rmse_test)