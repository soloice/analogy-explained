from __future__ import division
from question_words import QuestionWords
from embedding import Embedding
from learn_affine_transformation import AffineFitter
from sklearn.decomposition import PCA
import argparse
import numpy as np


parser = argparse.ArgumentParser()
parser.add_argument("--embedding_file", help="embedding file name",
                    default="../vectors-en/vec-text8-size50-win5-sg-hs.txt")
parser.add_argument("--analogy_file", help="analogy corpus file name", default="../data/questions-words.txt")
parser.add_argument("--learning_rate", help="learning rate", default=1e-2, type=float)
parser.add_argument("--l2_weight", help="l2 normalizer weight", default=1.0, type=float)
args = parser.parse_args()

emb = Embedding()
emb.load_embeddings(args.embedding_file, normalize=True)
qw = QuestionWords(args.analogy_file)
pairs = list(filter(lambda pair: (pair[0] in emb.word2id) and (pair[1] in emb.word2id),
                    qw.question_words_pairs["capital-world"]))
print(pairs)
words_x, words_y = zip(*pairs)
print(len(words_x), words_x)
print(len(words_y), words_y)
X = emb.extract_embeddings(words_x)
Y = emb.extract_embeddings(words_y)

def average_dist(M):
    m, n = M.shape
    dist = 0.0
    for i in range(m):
        for j in range(i+1, m):
            dist += np.linalg.norm(M[i, :] - M[j, :])
    # print(dist)
    return dist / (m * (m-1) / 2.0)

print(average_dist(X))
print(average_dist(Y))


def fit_with_model2(XY, low_dim=3):
    """ Fit X and Y uniformly. """
    # model = Model(n_components=3, random_state=0)
    model = PCA(n_components=low_dim)
    np.set_printoptions(suppress=True)
    num_samples2 = XY.shape[0] // 2  # The first num_samples2 samples are Xs, the rest are Ys.
    # y = [0] * num_samples2 + [1] * num_samples2
    XY_in_low_dim = model.fit_transform(XY)
    X_in_low_dim, Y_in_low_dim = XY_in_low_dim[:num_samples2, :], XY_in_low_dim[num_samples2:, :]
    # model.inverse_transform(X_in_low_dim)
    return model, X_in_low_dim, Y_in_low_dim


def test_fit_X_Y(X1, Y1, words_x1, words_y1, train_range=(0, -1), test_range=(0, -1),
                 l2_reg=0.0, remove_outliers=True, low_dim=20):
    if train_range[1] == -1:
        train_range = (train_range[0], len(X1))
    if test_range[1] == -1:
        test_range = (test_range[0], len(X1))

    X1_train, Y1_train = X1[train_range[0]:train_range[1], :], Y1[train_range[0]:train_range[1], :]
    XY = np.r_[X1_train, Y1_train]
    model, X_low_dim, Y_low_dim = fit_with_model2(XY, low_dim=low_dim)
    comp = model.components_  # shape = [n_components, n_features]

    af = AffineFitter()
    # Use all data (over-determined) to obtain (A_hat, b_hat) pair
    # A, b = af.fit(X_low_dim, Y_low_dim, learning_rate=5e-2, l2=l2_reg, my_eps=3e-7, max_iter=500000, verbose=5000)
    # b = b.ravel()
    A1, _ = af.fit(X_low_dim, Y_low_dim, learning_rate=5e-2, l2=l2_reg, my_eps=3e-7, max_iter=500000, verbose=5000)
    # print(A1.shape, comp.shape, np.linalg.pinv(comp).shape)
    A = np.dot(np.linalg.pinv(comp), np.dot(A1, comp))
    # print("A.shape = ", A.shape)
    fit_error = Y1_train.T - np.dot(A, X1_train.T)
    # print(fit_error.shape)
    b = np.mean(fit_error, axis=1)
    # print("b.shape = ", b.shape)
    print("Determinants of A_hat:", np.linalg.det(A))
    print("b_hat: ", b)
    diff = Y1 - X1
    average_offset = np.mean(diff, axis=0)
    if remove_outliers:
        # outlier_measure = np.sum((diff - average_offset)**2, axis=1)
        outlier_measure = np.sum((diff) ** 2, axis=1)
        indices_to_use = np.argsort(outlier_measure)[1:-1] # remove last outliers
        average_offset = np.mean(diff[indices_to_use], axis=0)

    def add_label(w):
        if w in words_x1:
            return w + "(x)"
        elif w in words_y1:
            return w + "(y)"
        else:
            return w

    top1_accuracy_affine, top3_accuracy_affine, identity_prediction_affine = 0, 0, 0
    top1_accuracy_offset, top3_accuracy_offset, identity_prediction_offset = 0, 0, 0
    # X1_test, Y1_test = X1[test_range[0]:test_range[1], :], Y1[test_range[0]:test_range[1], :]
    for i in range(test_range[0], test_range[1]):
        print(words_x1[i], words_y1[i])
        # prediction made by affine transformation
        x, y = X1[i], words_y1[i]
        y_pred1 = np.dot(A, x) + b
        # print(A.shape, x.shape, b.shape, y_pred1.shape)
        _, _, knn_words = emb.knn(y_pred1, k=10)
        if knn_words[0] == words_x1[i]:
            identity_prediction_affine += 1
        if knn_words[0] == y:
            top1_accuracy_affine += 1
            top3_accuracy_affine += 1
        if knn_words[1] == y or knn_words[2] == y:
            top3_accuracy_affine += 1
        knn_words = [add_label(w) for w in knn_words]
        print(knn_words)

        y_pred2 = x + average_offset
        _, _, knn_words = emb.knn(y_pred2, k=10)
        if knn_words[0] == words_x1[i]:
            identity_prediction_offset += 1
        if knn_words[0] == y:
            top1_accuracy_offset += 1
            top3_accuracy_offset += 1
        if knn_words[1] == y or knn_words[2] == y:
            top3_accuracy_offset += 1
        knn_words = [add_label(w) for w in knn_words]
        print(knn_words)
    print("Accuracy:")
    print("l2_reg: ", l2_reg)
    num_test_samples = test_range[1] - test_range[0]
    print("Affine: top1 = {0}, top3 = {1}, self = {2} out of {3}".format(top1_accuracy_affine, top3_accuracy_affine,
                                                                         identity_prediction_affine, num_test_samples))
    print("Offset: top1 = {0}, top3 = {1}, self = {2} out of {3}".format(top1_accuracy_offset, top3_accuracy_offset,
                                                                         identity_prediction_offset, num_test_samples))


print("All data (overfit):")
test_fit_X_Y(X, Y, words_x, words_y, [0, -1], [0, -1], l2_reg=0.0)
print("================================")
test_fit_X_Y(Y, X, words_y, words_x, [0, -1], [0, -1], l2_reg=0.0)

print("Separate train/test data:")
for l2_reg in [0.0, 0.001, 0.01, 0.1, 1.0]:
    print("********************************")
    test_fit_X_Y(X, Y, words_x, words_y, [0, 30], [30, -1], l2_reg)
    print("================================")
    test_fit_X_Y(Y, X, words_y, words_x, [0, 30], [30, -1], l2_reg)
