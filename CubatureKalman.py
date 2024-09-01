import numpy as np
import math
from scipy.linalg import cholesky, inv

def Tria(A):
    _, R = np.linalg.qr(A.T)
    return R.T

class CubatureKalmanFilter:

    def __init__(self, x0, p0, q0, r0, fx, hx, h):
        self.Q = q0
        self.R = r0
        self.P0 = p0
        self.x0 = x0
        self.hx = hx
        self.fx = fx
        self.n = x0.shape[0]
        self.sigmas_f = np.zeros((2 * len(x0), len(x0)))
        self.sigmas_h = np.zeros((2 * len(x0), len(r0)))
        self.z = np.array([[None]*1]).T
        self.K = 0
        self.h = h
        self.residual = 0
        self.S_predicted = cholesky(p0).T
        self.S_updated = cholesky(p0).T

    def predict(self, x, P):
        #шаг 1
        sigmas = np.zeros((2*self.n, self.n))
        S = cholesky(P).T
        U = S * np.sqrt(self.n)
        # шаг 2
        for i in range(self.n):
            sigmas[i] = x + U[i]
            sigmas[self.n+i] = x - U[i]
        # шаг 3
        for i in range(len(x) * 2):
            self.sigmas_f[i] = self.fx(sigmas[i])
        # шаг 4
        x_predicted = sum(self.sigmas_f, 0) / self.sigmas_f.shape[0]
        # шаг 5
        P_predicted = np.zeros((self.sigmas_f.shape[1], self.sigmas_f.shape[1]))
        Chi_predicted = np.zeros((self.sigmas_f.shape[0], self.sigmas_f.shape[1]))
        for i in range(self.sigmas_f.shape[0]):
            P_predicted += np.outer(self.sigmas_f[i], self.sigmas_f[i]) - np.outer(x_predicted, x_predicted)
            Chi_predicted[i] += self.sigmas_f[i] - x_predicted
        P_predicted *= 1 / self.sigmas_f.shape[0]
        P_predicted += self.Q
        Chi_predicted *= 1 / np.sqrt(self.sigmas_f.shape[0])
        Sq = cholesky(self.Q).T
        self.S_predicted = Tria(np.block([Chi_predicted.T, Sq]))
        return x_predicted, P_predicted

    def update(self, data, x, P):
        # шаг 1
        sigmas = np.zeros((2 * self.n, self.n))
        S = cholesky(P).T
        U = S * np.sqrt(self.n)
        # шаг 2
        for i in range(self.n):
            sigmas[i] = x + U[i]
            sigmas[self.n + i] = x - U[i]
        # шаг 3
        for i in range(2*len(x)):
            self.sigmas_h[i] = self.hx(sigmas[i])
        # шаг 4
        self.z = sum(self.sigmas_h, 0) / self.sigmas_h.shape[0]
        # шаг 5
        P_innovation = np.zeros((self.sigmas_h.shape[1], self.sigmas_h.shape[1]))
        for i in range(self.sigmas_h.shape[0]):
            P_innovation += np.outer(self.sigmas_h[i], self.sigmas_h[i]) - np.outer(self.z, self.z)
        P_innovation *= 1 / self.sigmas_h.shape[0]
        P_innovation += self.R
        # шаг 6
        P_cross = np.zeros((len(x), self.sigmas_h.shape[1]))
        for i in range(self.sigmas_h.shape[0]):
            P_cross += np.outer(sigmas[i], self.sigmas_h[i])/self.sigmas_h.shape[0] - np.outer(x, self.z)
        # шаг 7
        self.K = np.dot(P_cross, inv(P_innovation))
        # шаг 8-9
        self.residual = np.subtract(data, self.z)
        x_next = x + np.dot(self.K, self.residual)
        P_next = P - np.dot(self.K, P_innovation).dot(self.K.T)
        return x_next, P_next

    def predict_sqr(self, x):
        #шаг 1
        sigmas = np.zeros((2*self.n, self.n))
        U = self.S_updated * np.sqrt(self.n)
        # шаг 2
        for i in range(self.n):
            sigmas[i] = x + U[i]
            sigmas[self.n + i] = x - U[i]
        # шаг 3
        for i in range(len(x) * 2):
            self.sigmas_f[i] = self.fx(sigmas[i])
        # шаг 4
        x_predicted = sum(self.sigmas_f, 0) / self.sigmas_f.shape[0]
        # шаг 5
        Chi_predicted = np.zeros((self.sigmas_f.shape[0], self.sigmas_f.shape[1]))
        for i in range(self.sigmas_f.shape[0]):
            Chi_predicted[i] += self.sigmas_f[i] - x_predicted
        Chi_predicted *= 1 / np.sqrt(self.sigmas_f.shape[0])
        Sq = cholesky(self.Q).T
        self.S_predicted = Tria(np.block([Chi_predicted.T, Sq]))
        P_predicted = np.dot(self.S_predicted, self.S_predicted.T)
        return x_predicted, P_predicted

    def update_sqr(self, data, x, P):
        # шаг 1
        sigmas = np.zeros((2 * self.n, self.n))
        U = self.S_predicted * np.sqrt(self.n)
        # шаг 2
        for i in range(self.n):
            sigmas[i] = x + U[i]
            sigmas[self.n + i] = x - U[i]
        # шаг 3
        for i in range(2*len(x)):
            self.sigmas_h[i] = self.hx(sigmas[i])
        # шаг 4
        self.z = sum(self.sigmas_h, 0) / self.sigmas_h.shape[0]
        # шаг 5
        Sigma_innovation = np.zeros((self.sigmas_h.shape[0], self.sigmas_h.shape[1]))
        for i in range(self.sigmas_h.shape[0]):
            Sigma_innovation[i] += self.sigmas_h[i] - self.z
        Sigma_innovation = Sigma_innovation.T * (1 / np.sqrt(self.sigmas_h.shape[0]))
        Sr = cholesky(self.R).T
        S_innovation = Tria(np.block([Sigma_innovation, Sr]))
        # шаг 6
        Chi_cross = np.zeros((self.sigmas_f.shape[0], self.sigmas_f.shape[1]))
        for i in range(self.sigmas_h.shape[0]):
            Chi_cross[i] += sigmas[i] - x
        Chi_cross = Chi_cross.T * (1 / np.sqrt(self.sigmas_h.shape[0]))
        # шаг 7
        P_cross_sqr = np.dot(Chi_cross, Sigma_innovation.T)
        half_k = np.dot(P_cross_sqr, inv(S_innovation.T))
        self.K = np.dot(half_k, inv(S_innovation))
        # шаг 8-9
        self.residual = np.subtract(data, self.z)
        self.S_updated = Tria(np.block([Chi_cross - np.dot(self.K, Sigma_innovation), np.dot(self.K, Sr)]))
        x_next = x + np.dot(self.K, self.residual)
        P_next = np.dot(self.S_updated, self.S_updated.T)
        return x_next, P_next
    def filter(self, data):
        try:
            x_next = self.x0.copy()
            P_next = self.P0.copy()
            measurement_data = np.zeros_like(data)
            state_data = np.zeros((len(data), len(self.x0)))
            state_data[0] = self.x0
            measurement_data[0] = self.hx(x_next)
            for i in range(1, len(data)):
                x_pred, P_pred = self.predict(x_next, P_next)
                x_next, P_next = self.update(data[i], x_pred, P_pred)
                state_data[i] = x_next
                measurement_data[i] = self.hx(x_next)
        except:
            x_next = self.x0.copy()
            P_next = self.P0.copy()
            measurement_data = np.zeros_like(data)
            state_data = np.zeros((len(data), len(self.x0)))
            state_data[0] = self.x0
            measurement_data[0] = self.hx(x_next)
            for i in range(1, len(data)):
                x_pred, P_pred = self.predict_sqr(x_next)
                x_next, P_next = self.update_sqr(data[i], x_pred, P_pred)
                state_data[i] = x_next
                measurement_data[i] = self.hx(x_next)
        return state_data, measurement_data

class AdaptiveCubatureKalmanFilter(CubatureKalmanFilter):

    def __init__(self, x0, p0, q0, r0, fx, hx, h, alpha):
        super().__init__(x0, p0, q0, r0, fx, hx, h)
        self.Q0 = q0
        self.R0 = r0
        self.alpha = alpha

    def adapt_Ahlagi(self, data, x, P, P_pred, sqr_flag):
        sigmas = np.zeros((2 * self.n, self.n))
        if sqr_flag:
            S = self.S_updated
        else:
            S = cholesky(P).T
        U = S * np.sqrt(self.n)
        sigmas_adapt = np.zeros_like(self.sigmas_h)
        # шаг 2
        for i in range(self.n):
            sigmas[i] = x + U[i]
            sigmas[self.n + i] = x - U[i]
        # шаг 3
        for i in range(2 * len(x)):
            sigmas_adapt[i] = self.hx(sigmas[i])
        # шаг 4
        z = sum(sigmas_adapt, 0) / sigmas_adapt.shape[0]
        d = np.subtract(data, z)
        d1 = np.subtract(data, self.hx(x))
        d = d.reshape(d.size, 1)
        eps = self.residual.reshape(self.residual.size, 1)
        self.R = self.alpha * self.R + (1 - self.alpha) * (d @ d.T + np.nan_to_num(self.hx(np.nan_to_num(self.hx(P_pred)).T)))
        #self.R = self.alpha * self.R + (1 - self.alpha) * (d @ d.T + self.h @ P_pred @ self.h.T)
        if sqr_flag:
            try:
                cholesky(self.R)
            except:
                self.R = np.diag(np.diag(self.R))
                self.R[np.isinf(self.R)] = np.finfo(np.float64).max
        self.Q = self.alpha * self.Q + (1 - self.alpha) * (self.K @ eps @ eps.T @ self.K.T)
        if sqr_flag:
            try:
                cholesky(self.Q)
            except:
                self.Q = np.diag(np.diag(self.Q))
                self.Q[np.isinf(self.Q)] = np.finfo(np.float64).max


    def filter(self, data):
        try:
            x_next = self.x0.copy()
            P_next = self.P0.copy()
            measurement_data = np.zeros_like(data)
            state_data = np.zeros((len(data), len(self.x0)))
            state_data[0] = self.x0
            measurement_data[0] = self.hx(x_next)
            for i in range(1, len(data)):
                x_pred, P_pred = self.predict(x_next, P_next)
                x_next, P_next = self.update(data[i], x_pred, P_pred)
                self.adapt_Ahlagi(data[i], x_next, P_next,  P_pred, False)
                state_data[i] = x_next
                measurement_data[i] = self.hx(x_next)
        except:
            x_next = self.x0.copy()
            P_next = self.P0.copy()
            measurement_data = np.zeros_like(data)
            state_data = np.zeros((len(data), len(self.x0)))
            state_data[0] = self.x0
            measurement_data[0] = self.hx(x_next)
            for i in range(1, len(data)):
                x_pred, P_pred = self.predict_sqr(x_next)
                x_next, P_next = self.update_sqr(data[i], x_pred, P_pred)
                self.adapt_Ahlagi(data[i], x_next, P_next,  P_pred, True)
                state_data[i] = x_next
                measurement_data[i] = self.hx(x_next)
        return state_data, measurement_data

class RobustCubatureKalmanFilter(CubatureKalmanFilter):

    def __init__(self, x0, p0, q0, r0, fx, hx, h, omega):
        super().__init__(x0, p0, q0, r0, fx, hx, h)
        self.omega = omega

    def update(self, data, x, P):
        sigmas = np.empty((2 * self.n, self.n))
        S = cholesky(P).T
        U = S * np.sqrt(self.n)
        for i in range(self.n):
            sigmas[i] = x + U[i]
            sigmas[self.n + i] = x - U[i]
        for i in range(2 * len(x)):
            self.sigmas_h[i] = self.hx(sigmas[i])
        self.z = sum(self.sigmas_h, 0) / self.sigmas_h.shape[0]
        P_innovation = np.zeros((self.sigmas_h.shape[1], self.sigmas_h.shape[1]))
        for i in range(self.sigmas_h.shape[0]):
            P_innovation += np.outer(self.sigmas_h[i], self.sigmas_h[i]) - np.outer(self.z, self.z)
        P_innovation *= 1 / self.sigmas_h.shape[0]
        P_innovation += self.R
        P_cross = np.zeros((len(x), self.sigmas_h.shape[1]))
        for i in range((2 * len(x))):
            P_cross += np.outer(sigmas[i], self.sigmas_h[i]) - np.outer(x, self.z)
        self.residual = np.subtract(data, self.z)
        eps = self.residual.reshape(1, self.residual.size)
        L = math.exp(-(eps @ inv(self.R) @ eps.T / 2 * self.omega ** 2))
        B = np.nan_to_num(self.hx(np.nan_to_num(self.hx(P_innovation*L)).T)) + self.R
        self.K = self.hx(P_cross * L).T @ inv(B)
        x_next = x + np.dot(self.K, self.residual)
        P_next = P - np.dot(self.K, P_innovation).dot(self.K.T)

        return x_next, P_next

    def update_sqr(self, data, x, P):
        # шаг 1
        sigmas = np.zeros((2 * self.n, self.n))
        U = self.S_predicted * np.sqrt(self.n)
        # шаг 2
        for i in range(self.n):
            sigmas[i] = x + U[i]
            sigmas[self.n + i] = x - U[i]
        # шаг 3
        for i in range(2 * len(x)):
            self.sigmas_h[i] = self.hx(sigmas[i])
        # шаг 4
        self.z = sum(self.sigmas_h, 0) / self.sigmas_h.shape[0]
        # шаг 5
        Sigma_innovation = np.zeros((self.sigmas_h.shape[0], self.sigmas_h.shape[1]))
        for i in range(self.sigmas_h.shape[0]):
            Sigma_innovation[i] += self.sigmas_h[i] - self.z
        Sigma_innovation = Sigma_innovation.T * (1 / np.sqrt(self.sigmas_h.shape[0]))
        Sr = cholesky(self.R).T
        # шаг 6
        Chi_cross = np.zeros((self.sigmas_f.shape[0], self.sigmas_f.shape[1]))
        for i in range(self.sigmas_h.shape[0]):
            Chi_cross[i] += sigmas[i] - x
        Chi_cross = Chi_cross.T * (1 / np.sqrt(self.sigmas_h.shape[0]))
        # шаг 7
        self.residual = np.subtract(data, self.z)
        eps = self.residual.reshape(1, self.residual.size)
        L = math.exp(-(eps @ inv(self.R) @ eps.T / 2 * self.omega ** 2))
        B = np.nan_to_num(self.hx(np.nan_to_num(self.hx(P*L)).T)) + self.R
        self.K = np.nan_to_num(self.hx(P * L).T) @ inv(B)
        # шаг 8-9
        self.S_updated = Tria(np.block([Chi_cross - np.dot(self.K, Sigma_innovation), np.dot(self.K, Sr)]))
        x_next = x + np.dot(self.K, self.residual)
        P_next = np.dot(self.S_updated, self.S_updated.T)
        return x_next, P_next

class HybridCubatureKalmanFilter(AdaptiveCubatureKalmanFilter):

    def __init__(self, x0, p0, q0, r0, fx, hx, h, alpha, omega):
        super().__init__(x0, p0, q0, r0, fx, hx, h, alpha)
        self.omega = omega

    def update(self, data, x, P):
        sigmas = np.empty((2 * self.n, self.n))
        U = cholesky(P) * np.sqrt(self.n)
        for i in range(self.n):
            sigmas[i] = x + U[i]
            sigmas[self.n + i] = x - U[i]
        for i in range(2 * len(x)):
            self.sigmas_h[i] = self.hx(sigmas[i])
        self.z = sum(self.sigmas_h, 0) / self.sigmas_h.shape[0]
        P_innovation = np.zeros((self.sigmas_h.shape[1], self.sigmas_h.shape[1]))
        for i in range(self.sigmas_h.shape[0]):
            P_innovation += np.outer(self.sigmas_h[i], self.sigmas_h[i]) - np.outer(self.z, self.z)
        P_innovation *= 1 / self.sigmas_h.shape[0]
        P_innovation += self.R
        P_cross = np.zeros((len(x), self.sigmas_h.shape[1]))
        for i in range((2 * len(x))):
            P_cross += np.outer(sigmas[i], self.sigmas_h[i]) - np.outer(x, self.z)
        self.residual = np.subtract(data, self.z)
        eps = self.residual.reshape(1, self.residual.size)
        L = math.exp(-(eps @ inv(self.R) @ eps.T / 2 * self.omega ** 2))
        B = np.nan_to_num(self.hx(np.nan_to_num(self.hx(P_innovation * L)).T)) + self.R
        self.K = self.hx(P_cross * L).T @ inv(B)
        x_next = x + np.dot(self.K, self.residual)
        P_next = P - np.dot(self.K, P_innovation).dot(self.K.T)
        return x_next, P_next

    def update_sqr(self, data, x, P):
        # шаг 1
        sigmas = np.zeros((2 * self.n, self.n))
        U = self.S_predicted * np.sqrt(self.n)
        # шаг 2
        for i in range(self.n):
            sigmas[i] = x + U[i]
            sigmas[self.n + i] = x - U[i]
        # шаг 3
        for i in range(2 * len(x)):
            self.sigmas_h[i] = self.hx(sigmas[i])
        # шаг 4
        self.z = sum(self.sigmas_h, 0) / self.sigmas_h.shape[0]
        # шаг 5
        Sigma_innovation = np.zeros((self.sigmas_h.shape[0], self.sigmas_h.shape[1]))
        for i in range(self.sigmas_h.shape[0]):
            Sigma_innovation[i] += self.sigmas_h[i] - self.z
        Sigma_innovation = Sigma_innovation.T * (1 / np.sqrt(self.sigmas_h.shape[0]))
        Sr = cholesky(self.R).T
        # шаг 6
        Chi_cross = np.zeros((self.sigmas_f.shape[0], self.sigmas_f.shape[1]))
        for i in range(self.sigmas_h.shape[0]):
            Chi_cross[i] += sigmas[i] - x
        Chi_cross = Chi_cross.T * (1 / np.sqrt(self.sigmas_h.shape[0]))
        # шаг 7
        self.residual = np.subtract(data, self.z)
        eps = self.residual.reshape(1, self.residual.size)
        L = math.exp(-(eps @ inv(self.R) @ eps.T / 2 * self.omega ** 2))
        B = np.nan_to_num(self.hx(np.nan_to_num(self.hx(P*L)).T)) + self.R
        self.K = np.nan_to_num(self.hx(P * L).T) @ inv(B)
        # шаг 8-9
        self.S_updated = Tria(np.block([Chi_cross - np.dot(self.K, Sigma_innovation), np.dot(self.K, Sr)]))
        x_next = x + np.dot(self.K, self.residual)
        P_next = np.dot(self.S_updated, self.S_updated.T)
        return x_next, P_next

