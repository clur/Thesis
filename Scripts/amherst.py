'''file contains utility functions for dealing with amherst word data

       0            1            2                3
    "Token"     "Rating"    "TokenCount"   "RatingWideCount"
    "ages"          5           425            4769921

0. Token column: The token under consideration.
1. Rating: The star-rating. All the texts in all the corpora are marked with a star rating, 1 through 5 stars (1: lowest; 5: highest).
2. Token Count: The number of times the n-gram appeared in reviews with that row's star rating.
3. RatingWideCount: The total number of n-grams in texts in that rating category.
   the frequency of a word W appearing in a particular rating category R: divide TokenCount of W for R by the RatingWideCount for R.
'''

import codecs
import numpy as np
import pandas
import os;

print os.getcwd()


def get_word_ratings(fname):
    '''put words into rating categories by how frequently they are each of the ratings'''
    df = pandas.read_table(fname, sep=' ')
    df['Freq'] = df.TokenCount / df.RatingWideCount  # calculate the frequency
    # df=df.sort('Freq',ascending=False)
    df = df[df.TokenCount > 0]  # remove rows with no occurences in rating
    # df.to_csv('Data/amherst/uni/NEW_english-amazon-reviewfield-unigrams.frame', sep=' ')
    # divide into pos neg and neutral ranks
    # pos=df[df.Rating.isin([5,4])]
    # neut=df[df.Rating.isin([3])]
    # neg=df[df.Rating.isin([1,2])]
    tokens = df.Token.values
    print tokens


# look at grouping by


def get_word_prob(line):
    '''given a line in the format token,rating,tokencount,ratingwidecount calculate frequency of word being in rating'''
    pass


#how to represent data, should it be a class with object dictionary that has dictionary for each rating?
class Words(object):
    def __init__(self):
        self.dic = {}

    def build(self, fname):
        df = pandas.read_table(fname, sep=' ')
        df['Freq'] = df.TokenCount / df.RatingWideCount  #calculate the frequency
        df = df[df.TokenCount > 0]  #remove rows with no occurences in rating
        tokens = df.Token.values
        print tokens


if __name__ == "__main__":
    f = 'Data/amherst/uni/english-amazon-reviewfield-unigrams.frame'
    get_word_ratings(f)