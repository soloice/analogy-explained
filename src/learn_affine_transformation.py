from __future__ import division
import numpy as np

class AffineFitter(object):
    """Give two n-by-m matrices X and Y, where n is # of samples, m is # of features.
        Find a best (A, b) pair such that MSE of 1/n * np.linalg.norm(np.dot(A, X.T) + b) ** 2 is minimized.
    """

    def __init__(self):
        self.A, self.b = None, None

    def fit(self, X, Y, learning_rate=5e-3, l2=1.0, my_eps=1e-4, max_iter=100000, verbose=500):
        n, m = X.shape
        self.A = np.random.random((m, m))
        I = np.eye(m)
        self.b = np.zeros((m, 1))
        diff = self.A.dot(X.T) + self.b - Y.T
        print("=" * 100)
        print("l2 regularization: ", l2)
        loss = np.linalg.norm(diff) ** 2 / n + l2 * np.linalg.norm(self.A - I) ** 2
        print("loss = ", loss)
        previous_loss = loss + 1.0
        i = 0
        # when there is a significant decrease in loss function and # of iteration does not exceed a limit
        while loss < previous_loss - my_eps and i < max_iter:
            grad_A = 2.0 * diff.dot(X) / n + 2.0 * l2 * (self.A - I)
            grad_b = 2.0 * np.sum(diff, axis=1) / n
            self.A -= learning_rate * grad_A
            self.b -= learning_rate * grad_b.reshape((m, 1))
            previous_loss = loss
            diff = self.A.dot(X.T) + self.b - Y.T
            loss = np.linalg.norm(diff) ** 2 / n + l2 * np.linalg.norm(self.A - I) ** 2
            i += 1
            if i % verbose == 1:
                print("At iteration {0}, loss = {1}, diff = {2}".format(i, loss, np.linalg.norm(diff) ** 2 / n))
        print("At iteration {0}, loss = {1}, diff = {2}".format(i, loss, np.linalg.norm(diff) ** 2 / n))
        return self.A, self.b

    def self_test(self, n=16, m=10, noise_strength=1e-4, learning_rate=5e-2, l2_weight=0.0, eps=1e-5, max_iter=100000):
        X = np.random.random((n, m))
        A = np.random.random((m, m))
        b = np.random.random((m, 1))
        Y = (np.dot(A, X.T) + b).T
        for i in range(len(Y)):
            Y[i] += np.random.random(m) * noise_strength

        print("Test ONE:")
        A_hat, b_hat = self.fit(X, Y, learning_rate=learning_rate, l2=l2_weight, my_eps=eps, max_iter=max_iter)
        print("Determinants of A and A_hat:", np.linalg.det(A), np.linalg.det(A_hat))
        print("Relative error of A: ", np.linalg.norm(A - A_hat) / np.linalg.norm(A))
        print("Relative error of b: ", np.linalg.norm(b_hat - b) / np.linalg.norm(b))
        b, b_hat = b.ravel(), b_hat.ravel()
        print(b)
        print(b_hat)

if __name__ == "__main__":
    af = AffineFitter()
    af.self_test(n=16, m=10, noise_strength=0.0005, learning_rate=5e-2, l2_weight=0.0, eps=1e-7, max_iter=50000)
    af.self_test(n=103, m=50, noise_strength=1e-4, learning_rate=2e-2, l2_weight=0.0, eps=1e-7, max_iter=1000000)
