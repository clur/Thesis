import re

__author__ = 'claire'
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

import pandas
from collections import defaultdict
import random
import codecs

# how to represent data, should it be a class with object dictionary that has dictionary for each rating?
class Words(object):
    def __init__(self):
        self.rating_dict = defaultdict(list)  # keys are ratings, values lists of words
        self.word_dict = defaultdict(int)  # keys are words, values are ratings
        self.senti_word_dict = defaultdict()  # keys are words from sentinet, values are ratings

    def build_dict(self, cat=False):
        """make rating dictionary, k=1,2,3,4,5 v=[list of words with (most frequently) that rating
        :param cat: Boolean, set cat = True if you want 3 classes -1,0,1 (1,2:-1 3:0 4,5:1)
        """
        df = pandas.read_table('Data/amherst/uni/english-amazon-reviewfield-unigrams.frame', sep=' ')
        df['Freq'] = df.TokenCount / df.RatingWideCount  # calculate the frequency
        df = df[df.TokenCount > 0]  # remove rows with no occurences in rating
        print 'creating dictionary'
        if cat == True:  # 3 classes
            print '3 classes: -1,0,1 (neg,neut,pos)'
            df['Cat'] = df['Rating'].map({1: -1, 2: -1, 3: 0, 4: 1, 5: 1})  # convert from 5 ratings to 3 classes
            grouped = df.groupby('Token')
            for _, group in grouped:
                g = group[group.Freq == group.Freq.max()]  # g is row in df with highest freq
                token = g.Token.values[0]
                cat = g.Cat.values[0]
                self.rating_dict[cat].append(token)
        else:  # 5 classes
            print '5 classes: rating 1 to 5'
            grouped = df.groupby('Token')

            for _, group in grouped:
                g = group[group.Freq == group.Freq.max()]  # g is row in df with highest freq
                token = g.Token.values[0]
                rating = g.Rating.values[0]
                self.rating_dict[rating].append(token)

    def build_dict_senti(self):
        """
        similar to above amherst dict but using sentiwordnet
        dict[w]=[posscore,negscore,objscore]
        so far ignoring the POS

        :return:
        """
        df = pandas.read_table('Data/sentiwordnet/SentiWordNet_3.0.0_20130122.txt', sep='\t', header=26)
        df['ObjScore'] = 1 - (df.PosScore + df.NegScore)  # get the objective score
        df = df.drop('Gloss', 1)  # don't care about gloss for now
        # df[df.POS=='n'].describe() # give some information about different POS classes
        df['SynsetTerms'] = [re.sub('#\d+', '', t) for t in
                             df.SynsetTerms]  # take out the word sense, don't care for now
        df['SplitSyn'] = df.apply(lambda x: x['SynsetTerms'].split(' '),
                                  axis=1)  # split the terms into list, synsetterms column can contain more than one word that are close in meaning
        data = [(w, df.POS[i], df.PosScore[i], df.NegScore[i], df.ObjScore[i]) for i, j in enumerate(df.SplitSyn) for w
                in j]  # get rows for individual words
        new = pandas.DataFrame(data, columns=['W', 'POS', 'P', 'N', 'O'])
        grouped = new.groupby(['W', 'POS'])  # groups all senses of a word that are f.ex. noun
        for w, scores in grouped:
            self.senti_word_dict[w] = scores.mean().tolist()  # order is P,N,O
            # keys are form  ('beduin', 'n'), values [P,N,O]


    def inv_dict(self):
        """
        invert the rating dictionary so that key is word and value is rating
        :return:inverted dictionary
        """
        values = set(a for b in self.rating_dict.values() for a in b)
        self.word_dict = dict(
            (new_key, [key for key, value in self.rating_dict.items() if new_key in value][0]) for new_key in values)


    def generate_text(self, cat=False, wpl=2, lines=1):
        '''use self.dic to randomly generate file with random combinations of words

        :param outf: filename to write to
        :param cat: set true if 3 classes
        :param wpl: words per line
        :param lines: millions of lines

        '''
        with codecs.open('Data/generated/_' + str(lines) + 'mil', 'w', 'utf-8') as f:
            print 'writing permutations...'
            for i in range(lines * 1000000):
                if cat:

                    rate = random.randint(-1, 1)
                else:
                    rate = random.randint(1, 5)
                print 'rate', rate, 'len',
                print len(self.rating_dict[rate])
                x = random.sample(self.rating_dict[rate], wpl)
                f.write(' '.join(x) + '\n')


if __name__ == "__main__":
    W = Words()
    W.build_dicts()
    W.generate_text()