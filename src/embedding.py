import codecs
import numpy as np

class Embedding(object):
    def __init__(self):
        self.embedding_matrix = None
        self.word2id = None
        self.id2word = None
        self.vocabulary_size = -1
        self.embedding_size = -1
        self.embedding_norm2 = None

    def load_embeddings(self, embedding_file_name, normalize=False):
        # load embedding from file `embedding_file_name`
        print("Load embeddings from {0}, which may take a few seconds...".format(embedding_file_name))
        with codecs.open(embedding_file_name, encoding="utf-8") as f:
            n_word, n_dim = f.readline().split()
            self.vocabulary_size, self.embedding_size = int(n_word), int(n_dim)
            self.embedding_matrix = np.zeros((self.vocabulary_size, self.embedding_size))
            self.id2word, self.word2id = {}, {}

            for i, line in enumerate(f):
                components = line.split()
                token, vector = components[0], np.array([float(num) for num in components[1:]])
                self.word2id[token] = i
                self.id2word[i] = token
                self.embedding_matrix[i] = vector
            self.embedding_norm2 = np.sum(self.embedding_matrix ** 2, axis=1)
            print("Embedding norm2 shape: ", self.embedding_norm2.shape)
            # print("Embedding of </eos>", self.embedding_matrix[0])
            # print("Squared norm: ", self.embedding_norm2)

            if normalize:
                self.embedding_matrix /= np.reshape(np.sqrt(self.embedding_norm2), [-1, 1])
                self.embedding_norm2 = np.ones(self.embedding_norm2.shape)
            # print("Embedding of </eos>", self.embedding_matrix[0])
            # print("Squared norm: ", self.embedding_norm2)
        print("Finish loading.")

    def extract_embeddings(self, words):
        # words: a list/tuple of tokens
        # Each row is a word embedding
        res = np.zeros((len(words), self.embedding_size))
        for i, word in enumerate(words):
            res[i] = self.embedding_matrix[self.word2id[word]]
        return res

    def knn(self, query, k=5):
        # query can be a word (str), a word id (int),
        #  or an embedding (nparray, not necessarily appears in self.embedding_matrix)
        if type(query) == int:
            embedding = self.embedding_matrix[query]
        elif type(query) == str:
            embedding = self.embedding_matrix[self.word2id[query]]
        else:
            embedding = query

        norm2 = np.sum(embedding**2)
        # print(norm2.shape)
        inner_product = self.embedding_matrix.dot(embedding)
        # print(inner_product.shape)
        # print(self.embedding_norm2.shape)
        squared_distance = self.embedding_norm2 + norm2 - 2 * inner_product
        top_k_indices = np.argsort(squared_distance)[:k]
        return top_k_indices, squared_distance[top_k_indices], [self.id2word[word_id] for word_id in top_k_indices]

if __name__ == "__main__":
    emb = Embedding()
    emb.load_embeddings("../vectors-en/vec-text8-size50-win5-cbow-hs.txt", normalize=True)
    print(emb.extract_embeddings(["japan", "ottawa"]))
    print(emb.knn("japan"))
