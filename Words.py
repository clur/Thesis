import re

__author__ = 'claire'
'''
file contains utility functions for dealing with word data
methods to generate dictionaries from amherst and sentiwordnet datasets
build methods build the dictionaries from file
to_json methods write them to json
load methods (fast) loads the previously built dictionaries from json files
'''

import pandas
from collections import defaultdict
import random
import codecs
import csv
import json

# how to represent data, should it be a class with object dictionary that has dictionary for each rating?
class Words(object):
    def __init__(self):
        self.amherst_dict = defaultdict(list)  # keys are ratings, values lists of words
        self.sentiword_dict = defaultdict()  # keys are words from sentinet, values are ratings

    def build_dict_amherst(self, cat=True):
        """make rating dictionary, k=w, v=rating that word most frequently has
        :param cat: Boolean, set cat = False if you want 5 classes , cat default True, 3 ratings -1,0,1 (1,2:-1 3:0 4,5:1)
        """
        df = pandas.read_table('Data/amherst/uni/english-amazon-reviewfield-unigrams.frame', sep=' ')
        df['Freq'] = df.TokenCount / df.RatingWideCount  # calculate the frequency
        df = df[df.TokenCount > 0]  # remove rows with no occurences in rating
        print 'creating amherst dictionary'
        if cat == True:  # 3 classes
            print '3 classes: -1,0,1 (neg,neut,pos)'
            df['Cat'] = df['Rating'].map({1: -1, 2: -1, 3: 0, 4: 1, 5: 1})  # convert from 5 ratings to 3 classes
            grouped = df.groupby('Token')
            for _, group in grouped:
                g = group[group.Freq == group.Freq.max()]  # g is row in df with highest freq
                token = g.Token.values[0]
                cat = g.Cat.values[0]
                self.amherst_dict[cat].append(token)
        else:  # 5 classes
            print '5 classes: rating 1 to 5'
            grouped = df.groupby('Token')

            for _, group in grouped:
                g = group[group.Freq == group.Freq.max()]  # g is row in df with highest freq
                token = g.Token.values[0]
                rating = g.Rating.values[0]
                self.amherst_dict[rating].append(token)
        values = set(a for b in self.amherst_dict.values() for a in b)
        self.amherst_dict = dict(
            (new_key, [key for key, value in self.amherst_dict.items() if new_key in value][0]) for new_key in values)

    def build_dict_sentiword(self):
        """
        similar to above amherst dict but using sentiwordnet
        dict[(w,postag)]=(posscore,negscore,objscore) note: both key and value are tuples

        """
        print 'creating sentiword dictionary'
        df = pandas.read_table('Data/sentiwordnet/SentiWordNet_3.0.0_20130122.txt', sep='\t', header=26)
        df['ObjScore'] = 1 - (df.PosScore + df.NegScore)  # get the objective score
        df = df.drop('Gloss', 1)  # don't care about gloss for now
        # df[df.POS=='n'].describe() # give some information about different POS classes
        df['SynsetTerms'] = [re.sub('#\d+', '', t) for t in
                             df.SynsetTerms]  # take out the word sense # tags, don't care for now
        df['SplitSyn'] = df.apply(lambda x: x['SynsetTerms'].split(' '),
                                  axis=1)  # split the terms into list, synsetterms column can contain more than one word that are close in meaning
        data = [(w, df.POS[i], df.PosScore[i], df.NegScore[i], df.ObjScore[i]) for i, j in enumerate(df.SplitSyn) for w
                in j]  # get rows for individual words

        new = pandas.DataFrame(data, columns=['W', 'POS', 'P', 'N', 'O'])
        grouped = new.groupby(['W', 'POS'])  # groups all senses of a word that are f.ex. noun
        for w, scores in grouped:
            self.sentiword_dict[w] = tuple(scores.mean().tolist())  # order is P,N,O
            # entries are form  ('beduin', 'n'), values [P,N,O]


    def to_json_amherst(self, outf):
        '''
        writes the amherst dic to json outf
        '''
        json.dump(self.amherst_dict, open(outf, 'w'))

    def to_json_sentiword(self, outf):
        '''
        writes the sentiword dic to json outf
        (dump as items, key in this dict is tuple and json can't take tuples
        postprocess in load function)
        '''
        json.dump(self.sentiword_dict.items(), open(outf, 'w'))

    def load_amherst(self, inf):
        '''
        loads amherst dict from json file
        :param inf:
        '''
        self.amherst_dict = json.loads(open(inf).read())

    def load_sentiword(self, inf):
        '''
        loads sentiword dict from json file
        post processes to convert keys from strings back to tuples
        :param inf:
        '''
        self.sentiword_dict = dict(map(tuple, kv) for kv in json.loads(open(inf).read()))


    def to_csv_sentiword(self, outf):
        '''
        write the dict to .csv file for inspecting
        :param outf:
        :return:
        '''
        with codecs.open(outf, 'w', 'utf8') as f:
            writer = csv.writer(f)
            writer.writerow(['Word', 'POS', 'P', 'N', 'O'])
            for k, v in self.sentiword_dict.iteritems():
                print k, type(k)
                print v, type(v)
                print k + v
                writer.writerow(list(k) + list(v))

    # not in use
    def inv_dict(self):
        """
        invert the rating dictionary so that key is word and value is rating
        :return:inverted dictionary
        """
        values = set(a for b in self.amherst_dict.values() for a in b)
        self.word_dict = dict(
            (new_key, [key for key, value in self.amherst_dict.items() if new_key in value][0]) for new_key in values)


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
                print len(self.amherst_dict[rate])
                x = random.sample(self.amherst_dict[rate], wpl)
                f.write(' '.join(x) + '\n')


if __name__ == "__main__":
    W = Words()
    W.build_dict_amherst(cat=True)
    W.build_dict_sentiword()
    # W.generate_text()
    W.to_json_amherst('amherst_dict.json')
    W.to_json_sentiword('sentiword_dict.json')
    W.load_amherst('amherst_dict.json')
    W.load_sentiword('sentiword_dict.json')