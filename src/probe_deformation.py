from question_words import QuestionWords
from embedding import Embedding
from learn_affine_transformation import AffineFitter
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
emb.load_embeddings(args.embedding_file)
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

# A_hat, b_hat = af.fit(X, Y, learning_rate=learning_rate, l2=l2_weight, my_eps=eps, max_iter=max_iter)
af = AffineFitter()
# Use all data (over-determined) to obtain a best (A, b) pair
A_best, b_best = af.fit(X, Y, learning_rate=5e-2, l2=0.0, my_eps=3e-7, max_iter=500000, verbose=5000)
print("Determinants of A_best:", np.linalg.det(A_best))
print(b_best.ravel())

for l2_reg in [0.0, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0]:
    A1, b1 = af.fit(X[:30, :], Y[:30, :], learning_rate=5e-2, l2=l2_reg, my_eps=3e-7, max_iter=500000, verbose=5000)
    print("Determinants of A:", np.linalg.det(A1))
    # print(b1.ravel())
    print("Relative error of A:", np.linalg.norm(A1 - A_best) / np.linalg.norm(A_best))
    print("Relative error of b: ", np.linalg.norm(b_best - b1) / np.linalg.norm(b_best))
