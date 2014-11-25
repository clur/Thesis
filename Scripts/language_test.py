"""
takes text filename as 1st argument, utf8 encoded text file, samples separated by newline character
2nd argument is outf
python language_test.py inf outf
"""
from __future__ import division
import re
import enchant
import codecs
import sys


def tokenize(text):
    """
    return list of tokens from string text based on token pattern
    :param text: string
    :return:list
    """
    token_pattern = r"(?u)\b(?<!@)\w\w+\b\'*\w*"  # from sklearn.text.py, added ignore @username
    token_pattern = re.compile(token_pattern)
    return token_pattern.findall(text)


d = enchant.Dict("en_US")
# chkr = SpellChecker("en_US")

filetocheck = sys.argv[1]
data = open(filetocheck).readlines()
# tr_data = [' '.join(i.split('\t')[1:]) for i in data]
# tr_label = [i.split('\t')[0] for i in data]
# print tr_data[0]
# print tr_label[0]

for i in range(len(data)):
    english = 0
    tokens = tokenize(data[i])
    for t in tokens:
        try:
            if d.check(t):
                english += 1
        except:
            pass
    if english == 0 or english / len(tokens) < 0.3:
        continue
    else:
        print 'is english:', english / len(tokens), data[i], '\n'
        with open(sys.argv[2], 'a') as f:
            f.write(data[i])

