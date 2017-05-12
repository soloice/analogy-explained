from sklearn.manifold import TSNE, MDS, LocallyLinearEmbedding, Isomap
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt
from question_words import QuestionWords
from embedding import Embedding
from learn_affine_transformation import AffineFitter
import argparse


# X = np.array([[0, 0, 0], [0, 1, 1], [1, 0, 1], [1, 1, 1]])
# model = TSNE(n_components=2, random_state=0)
# np.set_printoptions(suppress=True)
# res = model.fit_transform(X)
# # plt.plot(res)
# x, y = res[:, 0], res[:, 1]
# plt.scatter(x, y)
# xmin, xmax = np.min(x), np.max(x)
# ymin, ymax = np.min(y), np.max(y)
# x_range, y_range = xmax - xmin, ymax - ymin
# plt.xlim(xmin - x_range/10.0, xmax + x_range/10.0)
# plt.ylim(ymin - y_range/10.0, ymax + y_range/10.0)
# plt.show()

parser = argparse.ArgumentParser()
parser.add_argument("--embedding_file", help="embedding file name",
                    default="../vectors-en/vec-text8-size100-win5-cbow-neg5.txt")
parser.add_argument("--analogy_file", help="analogy corpus file name", default="../data/questions-words.txt")
parser.add_argument("--learning_rate", help="learning rate", default=1e-2, type=float)
parser.add_argument("--l2_weight", help="l2 normalizer weight", default=1.0, type=float)
args = parser.parse_args()

emb = Embedding()
emb.load_embeddings(args.embedding_file)
qw = QuestionWords(args.analogy_file)

relation_types = ["capital-common-countries", "capital-world", "currency", "city-in-state", "family",
                  "gram1-adjective-to-adverb", "gram2-opposite", "gram3-comparative", "gram4-superlative",
                  "gram5-present-participle", "gram6-nationality-adjective", "gram7-past-tense",
                  "gram8-plural", "gram9-plural-verbs"]


def fit_with_model(Model, X, Y):
    """ Fit X and Y separately. """
    model = Model(n_components=2, random_state=2345)
    # model = Model(n_components=2)
    np.set_printoptions(suppress=True)
    res = model.fit_transform(X)
    x1, y1 = res[:, 0], res[:, 1]
    plt.scatter(x1, y1, c='r', marker='o')

    res = model.fit_transform(Y)
    x2, y2 = res[:, 0], res[:, 1]
    plt.scatter(x2, y2, c='b', marker='x')

    for i in range(len(x1)):
        plt.plot([x1[i], x2[i]], [y1[i], y2[i]])

    title = relation + "(Separate)"
    if Model == PCA:
        title = "PCA-" + title
    elif Model == MDS:
        title = "MDS-" + title
    elif Model == TSNE:
        title = "t-SNE-" + title
    plt.title(title)
    plt.show()


def fit_with_model2(Model, XY):
    """ Fit X and Y uniformly. """
    # model = Model(n_components=2, random_state=0)
    model = Model(n_components=2)
    np.set_printoptions(suppress=True)
    num_samples2 = XY.shape[0] // 2
    # y = [0] * num_samples2 + [1] * num_samples2
    res = model.fit_transform(XY)

    x1, y1 = res[:num_samples2, 0], res[:num_samples2, 1]
    x2, y2 = res[num_samples2:, 0], res[num_samples2:, 1]

    plt.scatter(x1, y1, c='r', marker='o')
    plt.scatter(x2, y2, c='b', marker='x')

    for i in range(num_samples2):
        plt.plot([x1[i], x2[i]], [y1[i], y2[i]])

    title = relation + "(Joint)"
    if Model == PCA:
        title = "PCA-" + title
    elif Model == MDS:
        title = "MDS-" + title
    elif Model == TSNE:
        title = "t-SNE-" + title
    plt.title(title)
    plt.show()

for relation in relation_types:
    pairs = list(filter(lambda pair: (pair[0] in emb.word2id) and (pair[1] in emb.word2id),
                        qw.question_words_pairs[relation]))
    print(relation)
    print(pairs)
    words_x, words_y = zip(*pairs)
    print(len(words_x), words_x)
    print(len(words_y), words_y)
    X = emb.extract_embeddings(words_x)
    Y = emb.extract_embeddings(words_y)
    XY = np.r_[X, Y]

    # Separately dimension reduction
    fit_with_model(MDS, X, Y)
    fit_with_model(TSNE, X, Y)
    fit_with_model(PCA, X, Y)

    # Jointly dimension reduction
    fit_with_model2(MDS, XY)
    fit_with_model2(TSNE, XY)
    fit_with_model2(PCA, XY)
