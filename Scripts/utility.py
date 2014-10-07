__author__ = 'claire'
import codecs
from sklearn.linear_model.stochastic_gradient import SGDClassifier as sgd
from sklearn.grid_search import GridSearchCV
from sklearn.pipeline import Pipeline
# from numpy.random import random_sample
from sklearn.naive_bayes import BernoulliNB as nb
from sklearn.feature_extraction.text import TfidfVectorizer as tf
from sklearn.linear_model import LogisticRegression as log
from sklearn.svm import LinearSVC as svm
import numpy as np
from sklearn.metrics import classification_report, f1_score
import gensim
import re
from collections import Counter
import Words


def write_targets_twitter(textfile):
    """
    convert labels text neg, neutral, pos to -1,0,1
    :param textfile:
    :use: write_targets_twitter('Data/twitter/twitter.test') #writes the target in format for scorer
    """
    text = codecs.open(textfile, 'r', 'utf8').readlines()
    target = [r.split('\t')[2] for r in text]
    map = {'positive': 1, 'neutral': 0, 'negative': -1}
    with open(textfile + '.true', 'w') as f:
        for t in target:
            f.write(str(map[t]))
            f.write('\n')


def write_targets_amazon(textfile):
    """
    convert labels text neg, neutral, pos to -1,0,1
    :param textfile:
    :use: write_targets_amazon('Data/amazon/review.test') #writes the target in format for scorer
    appends .true to filename
    """
    text = open(textfile).readlines()
    target = [int(r.split('\t')[0]) for r in text]
    map = {1: -1, 2: -1, 3: 0, 4: 1, 5: 1}
    with open(textfile + '.true', 'w') as f:
        for t in target:
            f.write(str(map[t]))
            f.write('\n')


def prob_sample(values, probabilities, size):
    """returns a *random* sample of length size of the values based on probabilities
    values=list of length N
    probabilities=list of length N
    size=integer
    """
    bins = np.add.accumulate(probabilities)
    return values[np.digitize(random_sample(size), bins)]


def target2int(target):
    """converts list of words in target array into list of numbers.
    0='positive', 1= 'neutral', 2= 'negative'
    """
    classes = list(set(target))
    print [[i, classes[i]] for i in range(len(classes))]
    new = []
    for i in target:
        for j in range(len(classes)):
            if i == classes[j]:
                new.append(j)
    return new


def target2tri(target):
    """
    target is list with possible values 1-5, convert to list with POS-0=4,5, NEUT-1=3
    and NEG-2=1,2
    :param target: list
    :return:list
    """
    mapping = {5: 0, 4: 0, 3: 1, 2: 2, 1: 2}
    return map(lambda x: mapping[x], target)


def load_twitter(fname):
    """
    open and read fname, split into data and target
    :param fname: filename to read
    :return: data, list and target, list
    """
    raw = codecs.open(fname, 'r', 'utf-8').readlines()  # load and split data into reviews
    target = [r.split('\t')[2] for r in raw]  # target is pos,neg,neutral
    data = [r.split('\t')[3] for r in raw]  # review text
    data = [d.lower().strip() for d in data]
    return data, target


def load_twitter_2class(fname):
    """
    open and read fname, split into data and target
    :param fname: filename to read
    :return: data, list and target, list
    """
    raw = codecs.open(fname, 'r', 'utf-8').readlines()  # load and split data into reviews
    raw = [r for r in raw if r.split('\t')[2] != u'neutral']
    target = [r.split('\t')[2] for r in raw]  # target is pos,neg,neutral
    data = [r.split('\t')[3] for r in raw]  # review text
    data = [d.lower().strip() for d in data]
    target = [t for t in target if t != 0]
    return data, target


def load_amazon(fname):
    """
    open and read fname, split into data and target
    :param fname: filename to read
    :return: data, list and target, list
    """
    raw = codecs.open(fname, 'r', 'utf8').readlines()  # load and split data into reviews
    # random.shuffle(raw)
    target = [int(float((r.split('\t')[0]))) for r in raw]  # target is pos,neg,neutral
    data = [r.split('\t')[1] for r in raw]  # title and review text
    return data, target


def print_top10(vectorizer, clf, class_labels):
    """Prints features with the highest coefficient values, per class"""
    feature_names = vectorizer.get_feature_names()
    s = ''
    for i, class_label in enumerate(class_labels):
        top10 = np.argsort(clf.coef_[i])[-10:]
        s += "%s: %s\n" % (class_label, ", ".join(feature_names[j] for j in top10))
    return s


def bow_clf_twitter(trainfile, testfile):
    '''read in train and test file and perform classification experiment with sklearn bow features and clf'''

    tr_data, tr_target = load_twitter(trainfile)
    te_data, te_target = load_twitter(testfile)
    mapping = {'negative': -1, 'neutral': 0, 'positive': 1}
    tr_target = [mapping[t] for t in tr_target]
    te_target = [mapping[t] for t in te_target]
    # print tr_data[0]
    vec = tf(ngram_range=(1, 1), max_df=0.9, min_df=0.01, norm=None, use_idf=False)  # basic tfidf vectorizer
    print vec
    print 'TFIDF FITTING'
    vec.fit(tr_data)
    print 'TFIDF FIT'
    print 'TFIDF TRANSFORMING'
    X_train = vec.transform(tr_data)
    X_test = vec.transform(te_data)
    print 'TRANSFORMED'
    clf = log(class_weight='auto')
    print clf
    clf.fit(X_train, tr_target)  # fit classifier to training data
    # x=clf.predict(X_test)
    # with open('twitter.train_predict','w') as f:
    # for i in x:
    # f.write(str(i))
    # f.write('\n')
    print 'data:\ntrain size: %s test size: %s' % (str(len(tr_target)), str(len(te_target)))
    print 'train set class representation:' + str(Counter(tr_target))
    print 'test set class representation: ' + str(Counter(te_target))
    print '\nclassifier\n----'
    print 'Accuracy on train:', clf.score(X_train, tr_target)
    print 'Accuracy on test:', clf.score(X_test, te_target)
    print '\nReport\n', classification_report(te_target, clf.predict(X_test))


def bow_clf_amazon(trainfile, testfile):
    '''read in train and test file and perform classification experiment with sklearn bow features and clf'''

    tr_data, tr_target = load_amazon(trainfile)
    te_data, te_target = load_amazon(testfile)
    mapping = {1: -1, 2: -1, 3: 0, 4: 1, 5: 1}
    tr_target = [mapping[t] for t in tr_target]
    te_target = [mapping[t] for t in te_target]
    # print tr_data[0]
    vec = tf(ngram_range=(1, 5), max_df=0.9, min_df=0.01)  # basic tfidf vectorizer
    print vec
    print 'TFIDF FITTING'
    vec.fit(tr_data)
    print 'TFIDF FIT'
    print 'TFIDF TRANSFORMING'
    X_train = vec.transform(tr_data)
    X_test = vec.transform(te_data)
    print 'TRANSFORMED'
    clf = log()
    print clf
    clf.fit(X_train, tr_target)  # fit classifier to training data
    # x=clf.predict(X_test)
    # with open('twitter.train_predict','w') as f:
    # for i in x:
    # f.write(str(i))
    #         f.write('\n')
    print 'data:\ntrain size: %s test size: %s' % (str(len(tr_target)), str(len(te_target)))
    print 'train set class representation:' + str(Counter(tr_target))
    print 'test set class representation: ' + str(Counter(te_target))
    print '\nclassifier\n----'
    print 'Accuracy on train:', clf.score(X_train, tr_target)
    print 'Accuracy on test:', clf.score(X_test, te_target)
    print '\nReport\n', classification_report(te_target, clf.predict(X_test))



def bow_clf_twitter_grid(trainfile, testfile):
    '''read in train and test file and perform classification experiment with sklearn bow features and clf'''

    tr_data, tr_target = load_twitter(trainfile)
    te_data, te_target = load_twitter(testfile)
    mapping = {'negative': -1, 'neutral': 0, 'positive': 1}
    tr_target = [mapping[t] for t in tr_target]
    te_target = [mapping[t] for t in te_target]
    # print tr_data[0]

    pipeline = Pipeline([
        ('tf', tf()),
        ('clf', nb()),
    ])

    params = {
        'tf__ngram_range': ((1, 1), (1, 2), (1, 3), (1, 4), (1, 5)),
        'tf__use_idf': (True, False),
        'tf__norm': ('l1', 'l2', None),
        # 'clf__class_weight':('auto',None)
    }
    # find the best parameters for both the feature extraction and the
    # classifier
    grid_search = GridSearchCV(pipeline, params, n_jobs=-1, verbose=1)

    print("Performing grid search...")
    print("pipeline:", [name for name, _ in pipeline.steps])
    print("parameters:")
    print params
    import time

    t0 = time.time()
    grid_search.fit(tr_data, tr_target)

    print("Best score: %0.3f" % grid_search.best_score_)
    print("Best parameters set:")
    best_parameters = grid_search.best_estimator_.get_params()
    for param_name in sorted(params.keys()):
        print("\t%s: %r" % (param_name, best_parameters[param_name]))


def tokenize(text):
    """
    return list of tokens from string text based on token pattern
    :param text: string
    :return:list
    """
    token_pattern = r"(?u)\b\w\w+\b\'*\w*"  # from sklearn.text.py
    token_pattern = re.compile(token_pattern)
    return token_pattern.findall(text)


def vectorize_gensim(data, modelname):
    """
    :param data: data is a list of strings, each string is a document
    :return: a sparse array of vectors representing word embeddings for words in word2vec model according to the modelname
    """
    model = gensim.models.Word2Vec.load(modelname)
    new_data = []
    # oovdic=defaultdict(int)
    blank = np.ndarray(model.layer1_size)  # TODO check this makes sense for comparisons
    for i in data:  # for each document
        new_vec = []
        for j in tokenize(i):  #for each word in the document
            try:
                new_vec.append(model[j])  #add the model representation of the word/ the embedding
            except:
                new_vec.append(blank)  # to ensure document vectors are same size
                # oovdic[j]+=1
        new_data.append(new_vec)
    # print oovdic
    return new_data

def avg_senti(text, word_dict):
    """
    as below using senti word dict values
    :param text: senti pos tagged text (first stanford, then use postag replace
    :param word_dict: sentiworddict, can be loaded with W.load_sentiword('sentiword_dict.json')
    :return: ? 0 if more postive, 1 if more negative, 2 if a tie an average over the 3 scores from sentiworddict (pos,neg,neutral)
    """
    # TODO figure out how to use senti word scores
    scores = []
    # print word_dict
    for i in tokenize(text.lower()):
        try:
            k = tuple(i.split('_'))
            val = word_dict[k]
            val = val[:2]  #ignore the objective score
            scores.append(val)
            print i, val
        except:
            pass
    if len(scores) > 0:
        avg = np.mean(scores, axis=0)
        print 'AVG:', avg
        if avg[0] == avg[1]:
            result = 2
        else:
            result = np.argmax(avg)
    else:
        result = 2  #all words are oov, assign to neutral
    return result


def avg_senti3(text, word_dict):
    """
    as below using senti word dict values
    :param text: senti pos tagged text (first stanford, then use postag replace
    :param word_dict: sentiworddict, can be loaded with W.load_sentiword('sentiword_dict.json')
    :return: ? 0 if more postive, 1 if more negative, 2 if a tie an average over the 3 scores from sentiworddict (pos,neg,neutral)
    """
    # TODO figure out how to use senti word scores
    scores = []
    # print word_dict
    for i in tokenize(text.lower()):
        try:
            k = tuple(i.split('_'))
            val = word_dict[k]
            val = val[:2]  #ignore the objective score
            scores.append(val)
            print i, val
        except:
            pass
    if len(scores) > 0:
        avg = np.mean(scores, axis=0)
        print 'AVG:', avg
        argmax = np.argmax(avg)
        if avg[0] == avg[1] or avg[np.argmax(avg)] < 0.1:  #lower than a threshold gets marked as neutral
            result = 2
        else:
            result = np.argmax(avg)
    else:
        result = 2  #all words are oov, assign to neutral
    return result


def write_targets_senti_avg(test_data, word_dict, outf):
    """
    write labels to file, calculated using amherst_score method
    use:  # write_targets_amherst_avg(te_data,W.amherst_dict,'Data/twitter/amherst.test.pred') # write labels based on averages calculated from amherst data
    """
    with open(outf, 'w') as f:
        for i in test_data:
            print i, ':'
            avg = avg_senti3(i, word_dict)

            map = {0: 1, 1: -1, 2: 0}  # tuple is ordered(pos,neg,neutral)
            avg = map[avg]
            print 'mapped:', avg
            print
            f.write(str(avg))
            f.write('\n')


def avg_amherst(text, word_dict):
    """
    calculates the average sentiment score of a text as the average over ratings of unigrams in text based on
    amherst rating table.
    :param fname: textfile name to be scored
    :param word_dict: Words().dic
    :return: the average rating over words in text
    """
    score = []
    for i in tokenize(text.lower()):
        try:
            score.append(word_dict[i])
            # print i,':',word_dict[i]
        except:
            pass

    # print score
    # print np.mean(score)
    # print text
    # print
    return np.mean(score)

def write_targets_amherst_avg(test_data, word_dict, outf):
    """
    write labels to file, calculated using amherst_score method
    use:  # write_targets_amherst_avg(te_data,W.amherst_dict,'Data/twitter/amherst.test.pred') # write labels based on averages calculated from amherst data
    """
    with open(outf, 'w') as f:
        for i in test_data:
            avg = avg_amherst(i, word_dict)
            avg = int(round(avg, 0))  # round the average to the nearest integer so it can be mapped to class labels
            map = {1: -1, 2: -1, 3: 0, 4: 1, 5: 1}
            avg = map[avg]  # map it to class values
            f.write(str(avg))
            f.write('\n')


def scorer(true, pred):
    """
    use classification report method on files, format is -1,0,1
    :param true: e.g. 'Data/twitter/twitter.test.true'
    :param pred: e.g. 'Data/twitter/amherst.test.pred'
    :return:
    """

    print 'TEST:', true
    print 'PRED:', pred
    true = open(true).readlines()
    pred = open(pred).readlines()
    print '\nReport\n'
    print classification_report(true, pred)
    print 'Macro F1', f1_score(true, pred)


def exp_avg_senti_amazon():
    '''
    Averages for senti
    first need to pos tag, follow instructions in notes
    :param f:
    :return:
    '''
    f = 'Data/amazon/rand5000reviews.txt'
    write_targets_amazon(f)
    W = Words.Words()
    W.load_sentiword('sentiword_dict.json')
    # print W.sentiword_dict
    postag_replace(f + '_POS-tagged')
    text = open(f + '_POS-tagged_senti').readlines()
    write_targets_senti_avg(text, W.sentiword_dict, f + '_senti_pred')
    scorer(f + '.true', f + '_senti_pred')


def exp_avg_senti_twitter():
    '''
    Averages for senti
    first need to pos tag, follow instructions in notes
    :param f:
    :return:
    '''
    f = 'Data/twitter/twitter.test'
    write_targets_twitter(f)
    W = Words.Words()
    W.load_sentiword('sentiword_dict.json')
    # print W.sentiword_dict
    postag_replace(f + '_POS-tagged')
    text = open(f + '_POS-tagged_senti').readlines()
    write_targets_senti_avg(text, W.sentiword_dict, f + '_senti_pred')
    scorer(f + '.true', f + '_senti_pred')


def exp_avg_amherst_amazon():
    '''
    Averages for amherst
    '''
    f = 'Data/amazon/rand5000reviews.txt'
    te_data, te_target = load_amazon(f)
    write_targets_amazon(f)  # writes the labels to file in amazon folder named .true
    W = Words.Words()
    W.load_amherst('amherst_dict_5.json')
    write_targets_amherst_avg(te_data, W.amherst_dict,
                              f + '_amherst_pred')  # write labels based on averages calculated from amherst data
    scorer(f + '.true', f + '_amherst_pred')


def exp_avg_amherst_twitter():
    '''
    Averages for amherst
    '''
    f = 'Data/twitter/twitter.test'
    te_data, te_target = load_twitter(f)
    write_targets_twitter(f)  # writes the labels to file in amazon folder named .true
    W = Words.Words()
    W.load_amherst('amherst_dict_5.json')
    write_targets_amherst_avg(te_data, W.amherst_dict,
                              f + '_amherst_pred')  # write labels based on averages calculated from amherst data
    scorer(f + '.true', f + '_amherst_pred')


if __name__ == "__main__":
    '''
    # --------------------------------------------------------------------------------------------------- #
    # # BOW baseline
    # tr_data, tr_target = load_twitter('Data/twitter/twitter.train')
    # tr_data, tr_target = load_amazon('Data/amazon/review10.train')
    # te_data, te_target = load_twitter('Data/twitter/twitter.test')
    # te_data, te_target = load_amazon('Data/amazon/review10.test')

    # --------------------------------------------------------------------------------------------------- #
    ## convert AMAZON data to 3
    # tr_target = target2tri(tr_target)
    # te_target=target2tri(te_target)
    # convert TWITTER data to int
    # tr_target=target2int(tr_target)
    # te_target = target2int(te_target)
    # bow_clf(tr_data, tr_target, te_data, te_target)

    # --------------------------------------------------------------------------------------------------- #
    ##embedding stuff
    ##review-uni_10.model is trained from unigrams, 10 million lines of words generated from amherst data
    # X_train=vectorize_gensim(tr_data,'Data/models/review-uni_10.model')
    # X_test=vectorize_gensim(te_data,'Data/models/review-uni_10.model')
    # print len(X_test), len(X_train)
    # print len(X_test[0]), len(X_train[0])
    # clf = log()
    # clf.fit(X_train, tr_target)  #fit classifier to training data
    # print '\nclassifier\n----'
    # print 'Accuracy on train:', clf.score(X_train, tr_target)
    # print 'Accuracy on test:', clf.score(X_test, te_target)
    # print '\nReport\n', classification_report(te_target, clf.predict(X_test))

    # --------------------------------------------------------------------------------------------------- #
    W = Words.Words()
    # W.build_dict_amherst()  #creates the rating dictionary
    # W.build_dict_sentiword()
    W.load_amherst('amherst_dict.json')
    # W.load_sentiword('sentiword_dict.json')
    # print W.amherst_dict
    # print W.sentiword_dict
    # #TODO some mapping of sentiworddict values to pos neut neg etc
    # ---------------------------------------------------------------------

    # Averages for Amherst

    #TODO write average amherst(twitter and amazon) as methods that just require a filename for re running later

    f = 'Data/amazon/rand5000reviews.txt'

    te_data, te_target = load_amazon(f)
    write_targets_amazon(f)  #writes the labels to file in amazon folder named .true
    write_targets_amherst_avg(te_data, W.amherst_dict,f)  # write labels based on averages calculated from amherst data

    # scorer('Data/twitter/twitter.test.true','Data/twitter/amherst.test.pred')

    true=''
    pred=''
    scorer(true,pred)




    # ---------------------------------------------------------------------


    '''
