import codecs
import numpy as np

class Embedding(object):
    def __init__(self):
        self.embedding_matrix = None
        self.word2id = None
        self.id2word = None
        self.vocabulary_size = -1
        self.embedding_size = -1

    def load_embeddings(self, embedding_file_name):
        # load embedding from file `embedding_file_name`
        print("Load embeddings, which may take a few seconds...")
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
        print("Finish loading.")

    def extract_embeddings(self, words):
        # words: a list/tuple of tokens
        # Each row is a word embedding
        res = np.zeros((len(words), self.embedding_size))
        for i, word in enumerate(words):
            res[i] = self.embedding_matrix[self.word2id[word]]
        return res

if __name__ == "__main__":
    emb = Embedding()
    emb.load_embeddings("../vectors-en/vec-text8-size50-win5-cbow-hs.txt")
    print(emb.extract_embeddings(["japan", "ottawa"]))
    print("Japan")
