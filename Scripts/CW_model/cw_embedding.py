import theano
import numpy as np
from theano import tensor as T
import cPickle
from collections import OrderedDict
from scipy import stats
import matplotlib.pyplot as plt
import time


class CW(object):
    def __init__(self, X, y, y_noise, V, K, num_context, n_hidden):
        """
        :param X: matrix of (B, H) input context word ids
        :param y: projection of target word
        :param V: size of vocabulary
        :param K: dimensionality of embeddings
        :param n_hidden: size of hidden layer
        """

        n_in = (num_context + 1) * K  # number of context words
        print 'n_in', n_in
        print 'n_hidden', n_hidden
        print 'Shapes:'
        print 'R:', V, 'x', K
        print 'W1:', n_in, 'x', n_hidden
        print 'W2:', n_hidden, 'x', 1

        randn = np.random.randn
        self.R = theano.shared(value=randn(V, K), name='R')
        self.hidden_bias = theano.shared(value=np.zeros((n_hidden,)), name='bias')
        self.W1 = theano.shared(value=randn(n_in, n_hidden), name='W1')
        self.W2 = theano.shared(value=randn(n_hidden, 1), name='W2')
        self.params = [self.R, self.W1, self.hidden_bias, self.W2]
        self.cost = self.get_cost(X, y, y_noise)

    def score(self, hiddens):
        return T.dot(hiddens, self.W2)


    def get_hidden(self, projections):
        pre_hidden = T.dot(projections, self.W1) + self.hidden_bias
        return T.nnet.sigmoid(pre_hidden)


    def get_projections(self, x, size):
        return self.R[x].reshape((size, (num_context + 1) * K), ndim=2)

    def concat_and_score(self, X, y):
        # Xy=T.concatenate([X,y.reshape(y.shape[0],1)], axis=1)
        Xy = T.concatenate([X, y], axis=1).flatten()
        projections = self.get_projections(Xy, X.shape[0])
        hiddens = self.get_hidden(projections)
        return T.mean(self.score(hiddens))  # scalar score


    def hinge_loss(self, s_pos, s_neg):
        return T.maximum(0, 1 - (s_pos - s_neg))

    def get_cost(self, X, y, y_noise):
        s_pos = self.concat_and_score(X, y)
        s_neg = self.concat_and_score(X, y_noise)
        return self.hinge_loss(s_pos, s_neg)


def gen_noise(distribution, shape0):
    """
    use the 'unigram' distribution to generate noise words from the vocabulary.
    the size will be (B,1)
    """
    return distribution.rvs(size=(shape0, 1))


start = time.time()
# load pickled context file, target file and vocab file
print 'loading data'
data = '/Users/claire/Dropbox/PycharmProjects/Thesis/Scripts/CW_model/'
words_x = cPickle.load(open(data + 'X.pickle', 'rb'))  # context words
words_y = cPickle.load(open(data + 'Y.pickle', 'rb'))  # target words
print 'loaded data'
print words_y[:5]
print words_y.shape
assert words_x.shape[0] == len(words_y)
vocab = cPickle.load(open(data + 'vocab.pickle', 'rb'))  # maps words to integer ids, most frequent word id=0 etc.
print 'loaded data'
# set sizes
K = 40  # embedding size
B = 10  # batchsize
n_hidden = K
num_context = words_x.shape[1]
V = len(vocab)


# create unigram distribution from top words file, used for noise generation
wfreq = open('top words.txt', 'r').readlines()
wfreq = [w.strip().split(':') for w in wfreq]
total = sum([int(w[1]) for w in wfreq])  # should this be the total number of tokens in the file??
dist = [(float(w[1]) / total) * 1.0 for w in wfreq]
unigram = stats.rv_discrete(name='unigram', values=(np.arange(len(dist)), dist))


# symbolic variables to pass to theano function
X = T.lmatrix(name='X2')
y = T.lmatrix(name='y2')
y_noise = T.lmatrix(name='ynoise')


# instantiate model object
model = CW(X, y, y_noise, V, K, num_context, n_hidden)

# Gradient descent
updates = OrderedDict()
lr = 1e-3  # learning rate
grads = T.grad(cost=model.cost,
               wrt=model.params)  # self.cost = self.get_cost(X, y, y_noise), self.params = [self.R, self.W1, self.hidden_bias, self.W2]
for p, g in zip(model.params, grads):
    updates[p] = p - lr * g

# train function
train = theano.function(inputs=[X, y, y_noise],
                        outputs=[model.cost],
                        updates=updates)

validate = theano.function(inputs=[X, y, y_noise],
                           outputs=[model.cost])  # no update step here


# actual loop that performs training
num_batches = words_x.shape[0] / B
for t in xrange(num_batches):
    # TODO add stopping criteria
    start_idx = t * B
    end_idx = (t + 1) * B
    print 'start %d, end %d' % (start_idx, end_idx)
    x_batch = words_x[start_idx:end_idx].astype('int32')
    y_batch = words_y[start_idx:end_idx].astype('int32')
    y_noise_batch = gen_noise(unigram, B)
    cost = train(x_batch, y_batch, y_noise_batch)[0]
    # if t % 100 == 0:
    if num_batches % 10 == 0:
        print "Batch: %d (of %d)/ cost = %.4f" % (t, num_batches, cost)
        with(open('validation.cost', 'a')) as f:  # write the validation cost for the whole set to file
            validation_cost = validate(words_x, words_y, gen_noise(unigram, len(words_y)))[0]
            f.write(str(validation_cost) + '\n')

print "took :", time.time() - start
# start 455080, end 455100
# Batch: 22754 / cost = 1.4434
# took : 27474.4183152

# save embeddings learned
inv_vocab = {v: k for k, v in vocab.items()}
# cPickle.dump(inv_vocab, open('inv_vocab.pickle', 'wb'))
with open('word_embeddings2.txt', 'w') as f:
    for i in inv_vocab.iterkeys():
        f.write(inv_vocab[i] + ' ')
        f.write(' '.join([str(r) for r in model.R.get_value()[i]]))
        f.write('\n')


def plot_validation(fname, batchsize):
    y = open('validation.cost').readlines()
    y = [i.strip() for i in y]
    y = [float(i) for i in y]
    x = range(len(y))
    x = [i * batchsize for i in x]
    print x
    plt.clf()
    plt.plot(x, y)
    plt.xlabel('num_batches')
    plt.ylabel('validation cost')
    plt.suptitle('Cost over total train set\nBatch size = %d' % batchsize)
    # plt.show()
    plt.savefig(fname)
