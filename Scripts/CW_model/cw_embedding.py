import codecs
import theano
import numpy as np
from theano import tensor as T
import cPickle
from collections import OrderedDict
from scipy import stats
import matplotlib.pyplot as plt
import time
from sklearn.metrics.pairwise import pairwise_distances as pd


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
        Xy = T.concatenate([X, y], axis=1).flatten()  # e.g. X=[the, cat, on, the] y=[sat]  Xy= [the, cat, on, the, sat]
        projections = self.get_projections(Xy, X.shape[0])
        hiddens = self.get_hidden(projections)
        return T.mean(self.score(hiddens))  # scalar score

    def hinge_loss(self, s_pos, s_neg):
        """
        Try this version of the hinge loss instead of T.maximum(0, 1 - positive_score - negative_score):
        scores = (T.ones_like(positive_score) - positive_score + negative_score)
        cost = (self.scores * (self.scores > 0)).mean()
        And then minimize cost. Let me know if that works or not.
        """
        scores = (T.ones_like(s_pos) - s_pos + s_neg)
        return (scores * (scores > 0)).mean()

    def hinge_loss_original(self, s_pos, s_neg):
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


def sanity_check(x, y, noise):
    """
    check the context and target make sense
    """
    # for batches, just check the first element of the batch
    x = x[0]
    y = y[0]
    noise = noise[0]
    context = [inv_vocab[i] for i in x]
    print 'target:', ' '.join(context[:2]) + ' [' + inv_vocab[y[0]] + '] ' + ' '.join(context[2:])
    print 'noise:', ' '.join(context[:2]) + ' [' + inv_vocab[noise[0]] + '] ' + ' '.join(context[2:])


start = time.time()
# load pickled context file, target file and vocab file
print 'loading data'
x = cPickle.load(open('X_.pickle', 'rb'))  # context words
y = cPickle.load(open('Y_.pickle', 'rb'))  # target words
split = int(0.90 * len(y))
words_x = x[:split]
words_y = y[:split]
validate_x = x[split:]  # first 10% for validation
validate_y = y[split:]
assert words_x.shape[0] == len(words_y)
vocab = cPickle.load(open('vocab.pickle', 'rb'))  # maps words to integer ids, most frequent word id=0 etc.
inv_vocab = {v: k for k, v in vocab.items()}  # inverted vocab for mapping back from indices to words
print 'loaded data'

# set sizes
K = 40  # embedding size
B = 20  # batchsize
n_hidden = K
num_context = words_x.shape[1]  # context size
V = len(vocab)  # vocab size

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

# validation function
validate = theano.function(inputs=[X, y, y_noise],
                           outputs=[model.cost])  # no update step here

check = []  # check the distance between related words 'good' and 'great'

num_batches = words_x.shape[0] / B
for t in xrange(num_batches):
    start_idx = t * B
    end_idx = (t + 1) * B
    x_batch = words_x[start_idx:end_idx].astype('int32')
    y_batch = words_y[start_idx:end_idx].astype('int32')
    y_noise_batch = gen_noise(unigram, B)
    # sanity_check(x_batch,y_batch,y_noise_batch)
    cost = train(x_batch, y_batch, y_noise_batch)[0]
    if t % 10 == 0:
        print "Batch: %d (of %d)/ cost = %.4f" % (t, num_batches, cost)
        with(open('validation.cost', 'a')) as f:  # write the validation cost for the whole set to file
            validation_cost = validate(validate_x, validate_y, gen_noise(unigram, len(validate_y)))[0]
            print "validation cost = %.4f" % validation_cost
            f.write(str(validation_cost) + '\n')
    if t % 100 == 0:  # check cosine distance
        dist = pd(model.R.get_value()[vocab['good']], Y=model.R.get_value()[vocab['great']], metric='cosine', n_jobs=1)[
            0]
        print '\ndist good and great', dist, '\n'
        check.append(dist)

print "took :", time.time() - start

# plot distance between related words 'good' and 'great'
plt.plot(range(len(check)), check)
plt.title('distance between "good" and "great"')
plt.xlabel('100*(batchsize=' + str(B))
plt.ylabel('cosine distance')
plt.show()


def plot_validation(fname, B):
    y = open('validation.cost').readlines()
    y = [i.strip() for i in y]
    y = [float(i) for i in y]
    x = range(len(y))
    x = [i * B for i in x]
    plt.clf()
    plt.plot(x, y)
    plt.xlabel('batches')
    plt.ylabel('validation cost')
    plt.suptitle('Cost over total train set\nBatch size = %d' % B)
    # plt.show()
    plt.savefig(fname)


plot_validation('valcost', B)

# save embeddings learned
with codecs.open('word_embeddings_temp.txt', 'w', 'utf8') as f:
    for i in inv_vocab.iterkeys():
        f.write(unicode(inv_vocab[i]) + ' ')
        f.write(' '.join([str(r) for r in model.R.get_value()[i]]))
        f.write('\n')

