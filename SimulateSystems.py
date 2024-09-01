import numpy as np


class StochasticNonlinearSystem:

    def __init__(self, initial_state, numsteps, theta_1, theta_2):
        self.x_init = initial_state
        self.N = numsteps
        self.theta_1 = theta_1
        self.theta_2 = theta_2

    def fx(self, x):
        x_new = np.zeros(3)
        x_new[2] += x[2]+1
        x_new[1] = x_new[2]/self.N
        x_new[0] = (1 - self.theta_2 / self.theta_1) * x[0] + 0.01 / self.theta_1 * (x_new[1] - x[0]) * np.exp(0.25 * (x_new[1] - x[0]))
        return x_new

    def hx(self, x):
        y_new = np.array([x[0]])
        return y_new
    def simulate(self, q = None, r = None):
        x = np.zeros((self.N, 3))
        y = np.zeros(self.N)
        x[0] = self.x_init
        for k in range(self.N-1):
            if q is not None:
                w = np.random.multivariate_normal(np.zeros(q.shape[0]), q)
                x[k + 1] = self.fx(x[k]) + [0.1 / self.theta_1 * w[0], w[1], w[2]]
            else:
                x[k + 1] = self.fx(x[k])

            if r is not None:
                v = np.random.multivariate_normal(np.zeros(r.shape[0]), r)
                y[k + 1] = self.hx(x[k + 1]) + v
            else:
                y[k + 1] = self.hx(x[k + 1])
        return y, x

class TargetTracking:

    def __init__(self , initial_state, numsteps, q1, q2, sr, so):
        self.x_init = initial_state
        self.N = numsteps
        self.T = 1.0
        self.Omega = np.deg2rad(initial_state[4])
        self.q1 = q1
        self.q2 = q2
        self.sr = sr
        self.so = so

    def fx(self, x):
        x_new = np.array([[1, np.sin(self.Omega*self.T)/self.Omega, 0.0, -((1-np.cos(self.Omega*self.T))/self.Omega), 0.0],
                          [0.0, np.cos(self.Omega*self.T), 0.0, -np.sin(self.Omega*self.T), 0.0],
                          [0.0, (1-np.cos(self.Omega*self.T))/self.Omega, 1, np.sin(self.Omega*self.T)/self.Omega, 0.0],
                          [1e-5, np.sin(self.Omega*self.T), 0.0, np.cos(self.Omega*self.T), 0.0],
                          [0.0, 0.0, 0.0, 0.0, 1]])
        return np.dot(x_new, x.T)

    def hx(self, x):
        y_new = np.array([np.sqrt(x[0]**2+x[2]**2),
                          np.arctan(x[2]/x[0])])
        return y_new

    def simulate(self, q=None, r=None):
        x = np.zeros((self.N, 5))
        y = np.zeros((self.N, 2))
        x[0] = self.x_init
        y[0] = self.hx(x[0])
        for k in range(self.N - 1):
            if q is not None:
                w = np.random.multivariate_normal(np.zeros(q.shape[0]), q)
                x[k + 1] = self.fx(x[k]) + w
            else:
                x[k + 1] = self.fx(x[k])

            if r is not None:
                v = np.random.multivariate_normal(np.zeros(r.shape[0]), r)
                y[k + 1] = self.hx(x[k + 1]) + v
            else:
                y[k + 1] = self.hx(x[k + 1])
        return y, x