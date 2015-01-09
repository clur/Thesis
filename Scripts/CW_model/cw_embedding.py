import codecs
import theano
import numpy as np
from theano import tensor as T
import cPickle
from collections import OrderedDict
from scipy import stats
import matplotlib.pyplot as plt
import time
# from sklearn.metrics.pairwise import pairwise_distances as pd
from scipy.spatial.distance import pdist
import random

dir = 'Preprocessed_window_5'


class CW(object):
    def __init__(self, V, K, num_context, n_hidden, R=None):
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
        if R:
            self.R = theano.shared(value=R, name='R')
        else:
            self.R = theano.shared(value=randn(V, K), name='R')

        self.hidden_bias = theano.shared(value=np.zeros((n_hidden,)), name='bias')
        self.W1 = theano.shared(value=randn(n_in, n_hidden), name='W1')
        self.W2 = theano.shared(value=randn(n_hidden, 1), name='W2')
        self.params = [self.R, self.W1, self.hidden_bias, self.W2]
        # self.cost = self.get_cost(X, y, y_noise)

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

    def hinge_loss_new(self, s_pos, s_neg):
        """
        Try this version of the hinge loss instead of T.maximum(0, 1 - positive_score - negative_score):
        scores = (T.ones_like(positive_score) - positive_score + negative_score)
        cost = (self.scores * (self.scores > 0)).mean()
        And then minimize cost. Let me know if that works or not.
        """
        scores = (T.ones_like(s_pos) - s_pos + s_neg)
        return (scores * (scores > 0)).mean()

    def hinge_loss(self, s_pos, s_neg):
        self.s_pos = s_pos
        self.s_neg = s_neg
        return T.maximum(0, 1 - (s_pos - s_neg))  # 1-(4 - 5)

    # def get_cost(self, X, y, y_noise):
    # s_pos = self.concat_and_score(X, y)
    # s_neg = self.concat_and_score(X, y_noise)
    # return self.hinge_loss(s_pos, s_neg)

    def get_cost_updates(self, X, y, y_noise, learning_rate):
        s_pos = self.concat_and_score(X, y)
        s_neg = self.concat_and_score(X, y_noise)
        cost = self.hinge_loss(s_pos, s_neg)
        gparams = T.grad(cost, self.params)
        updates = [(param, param - learning_rate * gparam) for param, gparam in zip(self.params, gparams)]
        learning_rate *= 0.95
        return (cost, updates)

    def get_cost(self, X, y, y_noise):
        s_pos = self.concat_and_score(X, y)
        s_neg = self.concat_and_score(X, y_noise)
        cost = self.hinge_loss(s_pos, s_neg)
        return cost


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
    context = [inv_vocab[w] for w in x]
    print 'target:', ' '.join(context[:2]) + ' [' + inv_vocab[y[0]] + '] ' + ' '.join(context[2:])
    print 'noise:', ' '.join(context[:2]) + ' [' + inv_vocab[noise[0]] + '] ' + ' '.join(context[2:])


# for debugging theano graph
def inspect_inputs(i, node, fn):
    print i, node, "input(s) value(s):\n", [input[0] for input in fn.inputs],


def inspect_outputs(i, node, fn):
    print "output(s) value(s):\n", [output[0] for output in fn.outputs]


theano.config.mode = 'FAST_RUN'
# theano.printing.pydotprint(train, outfile='graph_train_cw', var_with_name_simple=True)
# theano.printing.pydotprint(validate, outfile='graph_validate_cw', var_with_name_simple=True)

start = time.time()
# create unigram distribution from top words file, used for noise generation
text = open('all_vocab_no_single_occ.txt').readlines()[:30000]  # no of top words
wfreq = [w.strip().split('\t') for w in text]
total = sum([int(w[1]) for w in wfreq])  # should this be the total number of tokens in the file??
dist = [(float(w[1]) / total) * 1.0 for w in wfreq]
unigram = stats.rv_discrete(name='unigram', values=(np.arange(len(dist)), dist))

# load pickled context file, target file and vocab file
print 'loading data'
x = cPickle.load(open(dir + '/X.pickle', 'rb'))  # context words
y = cPickle.load(open(dir + '/Y.pickle', 'rb'))  # target words

print 'shuffling data'
combined = zip(x, y)
random.shuffle(combined)
x[:], y[:] = zip(*combined)

print 'splitting train and validation'
split = int(0.80 * len(y))
words_x = x[:split]
words_y = y[:split]
validate_x = x[split:]  # first 20% for validation
validate_y = y[split:]
# words_x = x
# words_y = y
assert words_x.shape[0] == len(words_y)
vocab = cPickle.load(open(dir + '/vocab.pickle', 'rb'))  # maps words to integer ids, most frequent word id=0 etc.
# inv_vocab = {v: k for k, v in vocab.items()}  # inverted vocab for mapping back from indices to words
# cPickle.dump(inv_vocab, open('inv_vocab.pickle', 'wb'))
inv_vocab = cPickle.load(open(dir + '/inv_vocab.pickle', 'rb'))
print 'loaded data'


def plot_all(e):
    fo = 'word_embeddings/'
    # plt.plot(range(len(t_costs)), t_costs)
    # # plt.ylim([-10,10])
    # plt.title('t_cost')
    # # plt.show()
    # plt.savefig(fo+'t_cost_'+str(e))
    # plt.clf()
    plt.plot(range(len(v_costs)), v_costs)
    plt.title('v_cost')
    # plt.show()
    plt.savefig(fo + 'v_cost_' + str(e))
    plt.clf()

    # plot distance between related words 'good' and 'great'
    plt.title('distance between "good" and "great"')
    plt.subplot(1, 2, 1)
    plt.plot(range(len(check1)), check1, label='euclidean')
    # plt.xlabel('1000 batches')
    # plt.ylabel('distance')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(range(len(check2)), check2, label='cosine')
    plt.legend()
    # plt.show()
    plt.savefig(fo + 'distance_' + str(e))
    plt.clf()

# set sizes
K = 40  # embedding size
B = 20  # batchsize
n_hidden = K
num_context = words_x.shape[1]  # context size
V = len(vocab)  # vocab size
print 'training samples:', "{:,}".format(len(words_x)), 'validation samples', "{:,}".format(len(validate_x))
print 'batchsize:', B
# symbolic variables to pass to theano function
X = T.lmatrix(name='X2')
y = T.lmatrix(name='y2')
y_noise = T.lmatrix(name='ynoise')
# load recent embedding to continue where training
# R = cPickle.load(open('R_0'))
# instantiate model object
model = CW(V, K, num_context, n_hidden)
lr = 1e-3  # learning rate
cost, updates = model.get_cost_updates(X, y, y_noise, learning_rate=lr)
v_cost = model.get_cost(X, y, y_noise)
train = theano.function(inputs=[X, y, y_noise], outputs=[cost, model.s_pos, model.s_neg],
                        updates=updates)  # , mode=theano.compile.MonitorMode(pre_func=inspect_inputs,post_func=inspect_outputs) )
validate = theano.function(inputs=[X, y, y_noise], outputs=[cost, model.s_pos,
                                                            model.s_neg])  # , mode=theano.compile.MonitorMode(pre_func=inspect_inputs,post_func=inspect_outputs))  # no update step here

check1 = []  # check the distance between related words 'good' and 'great'
check2 = []
v_costs = []
t_costs = []
epochs = 10

for e in range(epochs):
    print 'epoch ', e
    num_batches = words_x.shape[0] / B
    for t in xrange(num_batches):
        start_idx = t * B
        end_idx = (t + 1) * B
        x_batch = words_x[start_idx:end_idx].astype('int32')
        y_batch = words_y[start_idx:end_idx].astype('int32')
        y_noise_batch = gen_noise(unigram, B)

        c, s_pos, s_neg = train(x_batch, y_batch, y_noise_batch)
        t_costs.append(c)
        if c < 0:
            print 'NEGATIVE COST'
            print 'pos', s_pos, 'neg', s_neg, 'cost', c
            print

        if t % 100 == 0:  # check cosine distance
            valid_c, valid_s_pos, valid_s_neg = validate(validate_x.astype('int32'), validate_y.astype('int32'),
                                                         gen_noise(unigram, len(validate_y)))
            v_costs.append(valid_c)
            # print 'pos',valid_s_pos, 'neg',valid_s_neg
            print "Batch: %d (of %d)/ v_cost = %.4f" % (t, num_batches, valid_c)
            # print 'batch:%d of %d' % (t,num_batches)
            # print 'pos', s_pos, 'neg', s_neg, 'batch cost', c
            good = model.R.get_value()[vocab['good']]
            great = model.R.get_value()[vocab['great']]
            # print np.vstack((good, great)).shape
            dist1 = pdist(np.vstack((good, great)), 'euclidean')
            # dist2 = pd(good, Y=great, metric='cosine', n_jobs=1)[0]
            dist2 = pdist(np.vstack((good, great)), 'cosine')
            # print '\neuc dist good and great', dist1
            # print 'cos dist good and great', dist2
            check1.append(dist1)
            check2.append(dist2)
            # if t % 20000 == 0:
            # save embeddings learned
    print "took :", time.time() - start
    plot_all(e)
    with open('word_embeddings/epoch_' + str(e) + '.txt', 'w') as f:
        for i in inv_vocab.iterkeys():
            f.write(inv_vocab[i] + ' ')
            f.write(' '.join([str(r) for r in model.R.get_value()[i]]))
            f.write('\n')
    cPickle.dump(model.R.get_value(), open('word_embeddings/R_' + str(e), 'wb'))

print "took :", time.time() - start
# plot_all(e)
