__author__ = 'claire'
import gensim

model = gensim.models.Word2Vec.load_word2vec_format('word_embeddings.txt', binary=False)
amazon_model = gensim.models.Word2Vec.load_word2vec_format(
    '/Users/claire/Dropbox/PycharmProjects/Thesis/Scripts/embeddings/amazon.neg.vecs', binary=False)
# print model['man'], model['woman']

print model.most_similar('the')
print
print amazon_model.most_similar('the')
print model['the']
