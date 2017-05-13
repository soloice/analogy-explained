from __future__ import division
import numpy as np

class DiagonalFitter(object):
    """Give two n-by-m matrices X and Y, where n is # of samples, m is # of features.
        Find a best (d, b) pair such that 1/n * \Sum_i (d * X[i] + b - Y[i])^2 is minimized.
    """

    def __init__(self):
        self.d, self.b = None, None

    def fit(self, X, Y, learning_rate=5e-3, l2=1.0, my_eps=1e-4, max_iter=100000, verbose=500):
        n, m = X.shape
        self.d = np.random.random(m)
        I = np.ones(m)
        self.b = np.zeros(m)
        diff = self.d * X + self.b - Y
        print("=" * 100)
        print("l2 regularization: ", l2)
        loss = np.linalg.norm(diff) ** 2 / n + l2 * np.linalg.norm(self.d - I) ** 2
        print("loss = ", loss)
        previous_loss = loss + 1.0
        i = 0
        # when there is a significant decrease in loss function and # of iteration does not exceed a limit
        while loss < previous_loss - my_eps and i < max_iter:
            grad_d = 2.0 * np.sum(diff * X, axis=0) / n + 2.0 * l2 * (self.d - I)
            grad_b = 2.0 * np.sum(diff, axis=0) / n
            self.d -= learning_rate * grad_d
            self.b -= learning_rate * grad_b
            previous_loss = loss
            diff = self.d * X + self.b - Y
            loss = np.linalg.norm(diff) ** 2 / n + l2 * np.linalg.norm(self.d - I) ** 2
            i += 1
            if i % verbose == 1:
                print("At iteration {0}, loss = {1}, diff = {2}".format(i, loss, np.linalg.norm(diff) ** 2 / n))
        print("At iteration {0}, loss = {1}, diff = {2}".format(i, loss, np.linalg.norm(diff) ** 2 / n))
        return self.d, self.b

    def self_test(self, n=16, m=10, noise_strength=1e-4, learning_rate=5e-2, l2_weight=0.0, eps=1e-5, max_iter=100000):
        X = np.random.random((n, m))
        d = np.ones(m) + np.random.random(m) * noise_strength
        b = np.random.random(m)
        Y = d * X + b
        for i in range(len(Y)):
            Y[i] += np.random.random(m) * noise_strength

        print("Test:")
        d_hat, b_hat = self.fit(X, Y, learning_rate=learning_rate, l2=l2_weight, my_eps=eps, max_iter=max_iter)
        print("Relative error of d: ", np.linalg.norm(d_hat - d) / np.linalg.norm(d))
        print("Relative error of b: ", np.linalg.norm(b_hat - b) / np.linalg.norm(b))
        print(d)
        print(d_hat)
        print(b)
        print(b_hat)

if __name__ == "__main__":
    df = DiagonalFitter()
    # Test result for following configurations:
    df.self_test(n=16, m=10, noise_strength=0.0005, learning_rate=5e-2, l2_weight=0.0, eps=1e-7, max_iter=50000)

    df.self_test(n=103, m=50, noise_strength=1e-4, learning_rate=2e-2, l2_weight=0.0, eps=1e-7, max_iter=1000000)
