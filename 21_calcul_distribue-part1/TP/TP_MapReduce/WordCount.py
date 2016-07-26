#!/usr/bin/env python

import hadoopy
import re
from stemming.porter2 import stem
import numpy as np


class mapper(object):
    def __init__(self):
        self.words_stop = [line.rstrip('\n') for line in open('stop_words.txt')]

    def map(self,key, value):
        word_count = {}
        word_stemmed = [stem(word.group().lower()) for word in re.finditer(r"\w+",value,re.UNICODE) if word.group().lower() not in self.words_stop]
        for word in word_stemmed:
            if word:
                if word in word_count:
                    word_count[word]+=1
                else:
                    word_count[word] = 1
        for word in word_count:
            yield word, {'url':key,'tf':float(word_count[word])/len(word_stemmed)}
            #yield word,1

class reducer(object):
    def __init__(self):
        self.Ndocs = 27474.0
    def reduce(self,key, values):
        tfidf = {}
        values = list(values)
        for v in values:
            tfidf[v['url']] = v['tf']*np.log10(self.Ndocs/len(values))
        yield key,tfidf

if __name__ == "__main__":
    hadoopy.run(mapper, reducer, doc=__doc__)
