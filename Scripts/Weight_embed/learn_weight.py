__author__ = 'claire'

"""
Learn features weights using (polyglot) embedding representations of emoticon data as labeled data
"""
import numpy as np
import codecs
import re


def tokenize(text):
    """
    return list of tokens from string text based on token pattern
    :param text: string
    :return:list
    """
    token_pattern = r"(?u)\b\w\w+\b\'*\w*"  # from sklearn.text.py
    token_pattern = re.compile(token_pattern)
    return token_pattern.findall(text)


def tokenize():
    vec__token_pattern: "#*\\b(?<!@)\\w['\\-]*\\w+['\\-\\w]*\\b"


def vectorize(vecfile, data=[]):
    embeddings = open('cw_polyglot-64.vecs').readlines()
    labels = [i.split()[0] for i in embeddings]
    vecs = [np.array(map(float, i.split()[1:])) for i in embeddings]
    d = {}  # make embedding dictionary
    for i in range(len(labels)):
        d[labels[i]] = vecs[i]
    # print d
    for i in data:


vecfile = 'cw_polyglot-64.vecs'  # which embeddings to use
vectorize(vecfile)
emoticon = open('/Users/claire/Dropbox/PycharmProjects/Thesis/Scripts/Data/twitter_CST/englishtweets.txt').readlines()
target = [i.split('\t')[0] for i in emoticon]
data = [i.split('\t')[1:] for i in emoticon]

