from __future__ import division
import re
import enchant
import codecs


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

filetocheck = 'Data/twitter_CST/neg_100k.labeled'
data = open(filetocheck).readlines()
tr_data = [' '.join(i.split('\t')[1:]) for i in data]
tr_label = [i.split('\t')[0] for i in data]
print tr_data[0]
print tr_label[0]

for i in range(len(tr_data)):
    english = 0
    tokens = tokenize(tr_data[i])
    for t in tokens:
        try:
            if d.check(t):
                english += 1
        except:
            pass
    if english == 0 or english / len(tokens) < 0.5:
        continue
    else:
        print 'is english:', english / len(tokens), tr_data[i], '\n'
        # with open('englishtweets.both','a') as f:
        # f.write(tr_label[i]+'\t'+tr_data[i])

