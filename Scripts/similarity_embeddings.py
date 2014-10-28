import codecs
import numpy as np
from sklearn.metrics.pairwise import pairwise_distances as pd
from pprint import pprint

__author__ = 'claire'

fname = 'CW_model/word_embeddings_temp.txt'
# fname='/Users/claire/Dropbox/PycharmProjects/Thesis/Scripts/embeddings/embeddings-scaled.EMBEDDING_SIZE=50.txt'
# fname='/Users/claire/Dropbox/PycharmProjects/Thesis/Scripts/embeddings/sswe-u.txt'
data = codecs.open(fname, 'r', 'utf8').readlines()
data = [d.strip() for d in data]
vecs = [i.split()[1:] for i in data]
vocab = [d.split()[0] for d in data]
# print labels
d = dict((key, map(float, value)) for (key, value) in zip(vocab, vecs))


def most_similar(w, count=10):
    """
    return top count closets words based on euclidean distance
    :param w: the word to get distances from
    :return: list of tuples with top count shortest words and their distances
    """
    X = vecs[vocab.index(w)]
    dists = pd(X, Y=vecs, metric='cosine', n_jobs=1)[0]
    top = dists.argsort()[:count]  # ten shortest distances
    return zip(np.array(vocab)[top], dists[top])


pprint(most_similar('law'))
pprint(most_similar('time'))
pprint(most_similar('big'))
pprint(most_similar('child'))
pprint(most_similar('good'))