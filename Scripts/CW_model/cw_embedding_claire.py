__author__ = 'claire'

import numpy as np
import theano
from theano import tensor as T
import cPickle
from theano import pprint


h=2
k=40
train_x = cPickle.load(open('Xs.pickle','rb'))
train_y = cPickle.load(open('ys.pickle','rb'))
vocab=cPickle.load(open('vocab.pickle','rb'))
v=len(vocab)
print train_x.shape
#randomly initialize embeddings
R=theano.shared(np.random.rand(v,k), name='R')


#network
n_in=5*k
n_hidden=(5*k)/2
n_out=1


x=T.lmatrix(name='x') #contexts
y=T.lvector(name='y') #targets

# proj_x=R[x[0].flatten()] #one example

bias=theano.shared(np.zeros((x.shape[0],1)),name='bias')
W1=theano.shared(np.zeros((100,200)),name='W1')
W2=theano.shared(np.zeros((200,1)), name='W2')
score=T.dot(T.sigmoid(T.dot(W1,T.contatenate(x,y))+bias),W2)

def score(X,y):
    return T.dot(theano.tensor.nnet.sigmoid(T.dot(W1,T.contatenate(X,y))+bias),W2)

def hinge_loss(s,s_c):
    return T.maximum(0,1-(s+s_c))



#projection layer

#projection layer
# idx=T.lscalar()
# x=T.lmatrix(name='x') #contexts
# y=T.lvector(name='y')
# proj_function=theano.function([X],T.concatenate([X[idx].flatten()],R[y[idx]]))

idx=0
proj_X=theano.shared(R[train_x[idx].flatten()])

train_model=theano.function(inputs=[idx],
                            outputs=score(X,y),
                            givens={
                                x:[train_x[idx*batch_size:(index+1)*batch_size]],
                                y:[train_x[idx*batch_size:(index+1)*batch_size]])})
print train_model
# y_c= corrupt y

# s=score(proj_X,y)
# s_c=score(proj_X,y_c)
