# -*- coding: utf-8 -*-
import heapq
import numpy as np


def ugly_normalize(vecs):
    normalizers = np.sqrt((vecs * vecs).sum(axis=1))
    normalizers[normalizers == 0] = 1
    return (vecs.T / normalizers).T


class Embeddings(object):
    def __init__(self, vecsfile, normalize=True):
        self._vecs = None
        self._vocab = []
        with open(vecsfile, 'r') as f:
            vecs = []
            for l in f:
                line = l.strip().split()
                self._vocab.append(line[0])
                vecs.append([float(x) for x in line[1:]])
            self._vecs = np.array(vecs)
        if normalize:
            self._vecs = ugly_normalize(self._vecs)
            self._w2v = {w: i for i, w in enumerate(self._vocab)}

    @classmethod
    def load(cls, vecsfile):
        return Embeddings(vecsfile)

    def word2vec(self, w):
        return self._vecs[self._w2v[w]]

    def similar_to_vec(self, v, N=10):
        sims = self._vecs.dot(v)
        sims = heapq.nlargest(N, zip(sims, self._vocab, self._vecs))
        return sims

    def similarity(self, w1, w2):
        v1 = self.word2vec(w1)
        v2 = self.word2vec(w2)
        return v1.dot(v2)

    def most_similar(self, word, N=10):
        w = self._vocab.index(word)
        sims = self._vecs.dot(self._vecs[w])
        sims = heapq.nlargest(N, zip(sims, self._vocab))
        return sims


if __name__ == '__main__':
    import sys

    e = Embeddings.load(sys.argv[1])
    w1 = sys.argv[2] if len(sys.argv) > 2 else 'man'
    w2 = sys.argv[3] if len(sys.argv) > 3 else 'woman'
    print('computing similarity between "{}" and "{}"'.format(w1, w2))
    print(e.similarity(w1, w2))
