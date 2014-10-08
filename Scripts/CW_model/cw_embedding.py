import theano
import numpy as np
from theano import tensor as T
import cPickle
from collections import OrderedDict

class CW(object):
    def __init__(self, X, y, y_noise, V, K, num_context, n_hidden, unigram):
        """
        :param X: matrix of (B, H) input context word ids
        :param y: projection of target word
        :param V: size of vocabulary
        :param K: dimensionality of embeddings
        :param n_hidden: size of hidden layer
        :param unigram: (V,1)-vector of unigram word probabilities
        """

        n_in = (num_context + 1) * K  # number of context words
        randn = np.random.randn
        self.unigram = unigram
        self.R = theano.shared(value=randn(V, K), name='R')
        self.hidden_bias = theano.shared(value=np.zeros((n_hidden,)), name='bias')
        # print (n_in, n_hidden)
        self.W1 = theano.shared(value=randn(n_in, n_hidden), name='W1')
        self.W2 = theano.shared(value=randn(n_hidden, 1), name='W2')
        self.params = [self.R, self.W1, self.hidden_bias, self.W2]
        self.cost = self.get_cost(X, y, y_noise)

    def score(self, hiddens):
        return T.dot(hiddens, self.W2)

    def get_projections(self, x):
        return self.R[x.flatten()]

    def get_hidden(self, projections):
        pre_hidden = T.dot(projections, self.W1) + self.hidden_bias
        return T.nnet.sigmoid(pre_hidden)

    def concat_and_score(self, X, y):
        batch_size, context_size = X.shape  # e.g(45000,4)
        embedding_size = self.R.shape[1]
        contexts = self.get_projections(X).reshape((batch_size, context_size * embedding_size))
        targets = self.get_projections(y)
        projections = T.concatenate([contexts, targets], axis=1)
        hiddens = self.get_hidden(projections)
        return T.mean(self.score(hiddens))  # scalar score

    # def concat_and_score(self, X, y):
    # Xy=T.horizontal_stack(X,y)
    #     projections = self.get_projections(Xy)
    #     hiddens = self.get_hidden(projections)
    #     return T.mean(self.score(hiddens)) # scalar score


    def hinge_loss(self, s_pos, s_neg):
        return T.maximum(0, 1 - (s_pos - s_neg))

    def get_cost(self, X, y, y_noise):
        s_pos = self.concat_and_score(X, y)
        s_neg = self.concat_and_score(X, y_noise)
        return self.hinge_loss(s_pos, s_neg)

K = 40
B = 10

data = '/Users/claire/Dropbox/PycharmProjects/Thesis/Data/CW_data/'

n_hidden = K
words_x = cPickle.load(open(data + 'Xs.pickle', 'rb'))
words_y = cPickle.load(open(data + 'ys.pickle', 'rb'))
vocab = cPickle.load(open(data + 'vocab.pickle', 'rb'))

num_context = words_x.shape[1]
print 'num_context', num_context
V = len(vocab)
unigram = np.ones((V,1)) / (1.*V)

# print 'words_x shape:', words_x.shape
# print 'words_y shape:', words_y.shape
# print 'len vocab:', V

X = T.lmatrix()
y = T.lvector()
y_noise = T.lvector()

model = CW(X, y, y_noise, V, K, num_context, n_hidden, unigram)

updates = OrderedDict()
lr = 1e-3
grads = T.grad(model.cost, model.params)
for p,g in zip(model.params, grads):
    updates[p] = p - lr * g

train = theano.function([X, y, y_noise],
                        [model.cost],
                        updates=updates)
def gen_noise():
    return np.random.randint(low=0, high=V, size=(B, 1))

# num_batches = words_x.shape[0] / B
# for t in xrange(num_batches):
for t in xrange(2):
    start_idx = t * B
    end_idx = (t+1) * B
    x_batch = words_x[start_idx:end_idx]
    y_batch = words_y[start_idx:end_idx]
    y_noise_batch = gen_noise()[0]
    print x_batch.shape
    print y_batch.shape
    #print y_noise.shape
    cost = train(x_batch, y_batch, y_noise_batch)[0]
    print "Batch: %d / cost = %.4f" % (t, cost)

